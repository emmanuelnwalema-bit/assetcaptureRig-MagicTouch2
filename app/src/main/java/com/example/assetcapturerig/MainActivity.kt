package com.example.assetcapturerig

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.graphics.*
import android.media.AudioManager
import android.media.Image
import android.media.ToneGenerator
import android.opengl.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Base64
import android.util.Log
import android.view.PixelCopy
import android.view.SurfaceView
import android.view.View
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.Anchor
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import io.github.sceneview.ar.ARSceneView
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.*

class MainActivity : AppCompatActivity() {

    private val modalEndpoint = "https://emmanuelnwalema--mobile-6dof-capture-ui.modal.run/upload_plate"
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .writeTimeout(20, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .build()

    private lateinit var rootLayout: FrameLayout
    private lateinit var sceneView: ARSceneView
    private lateinit var overlayView: PrismOverlayView
    private lateinit var snapFlash: View

    private lateinit var targetBadge: TextView
    private lateinit var chkListLeft: TextView
    private lateinit var chkListRight: TextView
    private lateinit var valDist: TextView
    private lateinit var valAlign: TextView
    private lateinit var valScale: TextView

    private lateinit var distIndicator: TextView
    private lateinit var normalIndicator: TextView
    private lateinit var centerIndicator: TextView

    private lateinit var scaleBar: LinearLayout
    private lateinit var scaleReadout: TextView
    private lateinit var btnScaleDown: Button
    private lateinit var btnScaleUp: Button
    private lateinit var forceSnapBtn: Button
    private lateinit var batchUploadBtn: Button
    private lateinit var actionButton: Button

    private lateinit var previewScrollView: HorizontalScrollView
    private lateinit var previewContainer: LinearLayout

    private lateinit var fullPreviewLayout: FrameLayout
    private lateinit var fullImageView: ImageView
    private lateinit var fullPreviewLabel: TextView
    private lateinit var closePreviewBtn: Button

    private val debugLog = StringBuilder("App Initialized (60 FPS ARCore + Pause Auto-Snap MagicTouch)\n")
    private var toneGen: ToneGenerator? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    // =========================================================
    // SINGLE-THREADED NON-BLOCKING MAGICTOUCH ENGINE (CPU-XNNPACK)
    // =========================================================
    private val mlExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    @Volatile private var magicTouchInterpreter: Interpreter? = null
    private val magicTouchAssetName = "magic_touch.tflite"

    // Pre-allocated reusable buffers (accessed strictly on mlExecutor)
    private var mtInH = 512
    private var mtInW = 512
    private var mtInC = 4
    private var mtInType = DataType.FLOAT32
    private var mtOutShape = intArrayOf(1, 512, 512, 1)
    private var mtOutType = DataType.FLOAT32
    private var mtInputBuf: ByteBuffer? = null
    private var mtOutputBuf: ByteBuffer? = null
    private var mtPixelBuf: IntArray? = null
    private var mtFloatInput: FloatArray? = null
    private var mtFloatOutput: FloatArray? = null
    private var centerGaussMap: FloatArray? = null

    @Volatile private var isAutoSnapRunning = false
    @Volatile private var lastSnapLatencyMs = 0L
    private var preLockCopyBmp: Bitmap? = null

    // =========================================================
    // 60 FPS DWELL DETECTOR & WORLD-PINNED PRE-LOCK ANCHOR
    // =========================================================
    private val dwellDurationMs = 160L
    private var hasDwellRef = false
    private var dwellStartMs = 0L
    private var dwellRefX = 0f
    private var dwellRefZ = 0f
    private var dwellFwdX = 0f
    private var dwellFwdY = 0f
    private var dwellFwdZ = -1f
    private var dwellProgress = 0f

    // Prevents repeatedly firing MagicTouch on the same empty table spot
    private var hasFailedSpot = false
    private var failedSpotX = 0f
    private var failedSpotZ = 0f
    private var failedSpotTimeMs = 0L

    // Live camera pose for stale-frame rejection during auto-snap
    @Volatile private var liveCamX = 0f
    @Volatile private var liveCamY = 0f
    @Volatile private var liveCamZ = 0f
    @Volatile private var liveCamFwdX = 0f
    @Volatile private var liveCamFwdY = 0f
    @Volatile private var liveCamFwdZ = -1f

    private var currentStepIdx = 0
    private var isBoxPlaced = false
    private var isCapturing = false
    private var isBatchUploading = false
    private var lastFrame: Frame? = null
    private var preLockAnchor: Anchor? = null
    private var prismAnchor: Anchor? = null
    private var trackedTablePlane: Plane? = null
    private var initialAzimuth = 0f
    private var alignStartTime: Long? = null
    private var cooldownUntil: Long = 0L
    private var lastHudUpdateMs: Long = 0L

    // Adaptive Anchor-Translation Damper (absorbs vibration micro-jitter with 0ms camera lag)
    private var hasStableAnchor = false
    private var stableAnchorX = 0f
    private var stableAnchorY = 0f
    private var stableAnchorZ = 0f

    // Reusable per-frame camera matrices
    private val frameViewMatrix = FloatArray(16)
    private val frameProjMatrix = FloatArray(16)
    private val frameVpMatrix = FloatArray(16)

    private var prismWidth = 0.10f
    private var prismDepth = 0.10f
    private var prismHeight = 0.05f

    // =========================================================
    // WORLD-PINNED PRE-LOCK RING STATE (SET BY AUTO-SNAP MAGICTOUCH)
    // =========================================================
    @Volatile private var hasLiveCling = false
    @Volatile private var liveWorldX = 0f
    @Volatile private var liveWorldY = 0f
    @Volatile private var liveWorldZ = 0f
    @Volatile private var liveWidth = 0.10f
    @Volatile private var liveDepth = 0.10f
    @Volatile private var liveHeight = 0.045f
    @Volatile private var liveAzimuth = 0f

    private val capturedThumbnails = mutableMapOf<Int, Bitmap>()
    private val previewCardViews = mutableListOf<ImageView>()

    data class SequenceStep(
        val id: String,
        val label: String,
        val hint: String,
        val targetTilt: Int,
        val targetOrbit: Int
    )

    private val sequence = listOf(
        SequenceStep("front", "1. FRONT (0° LEVEL)", "AIM LEVEL AT FRONT", 0, 0),
        SequenceStep("hero_3_4", "2. HERO SEED (45° / 30° TILT)", "ORBIT 45° & TILT 30° DOWN", 30, 45),
        SequenceStep("right", "3. RIGHT PROFILE (0° LEVEL)", "AIM LEVEL AT RIGHT", 0, 90),
        SequenceStep("back_r", "4. REAR-RIGHT (30° TILT)", "ORBIT 135° & TILT 30° DOWN", 30, 135),
        SequenceStep("back", "5. BACK VIEW (0° LEVEL)", "AIM LEVEL AT REAR", 0, 180),
        SequenceStep("back_l", "6. REAR-LEFT (30° TILT)", "ORBIT 225° & TILT 30° DOWN", 30, 225),
        SequenceStep("left", "7. LEFT PROFILE (0° LEVEL)", "AIM LEVEL AT LEFT", 0, 270),
        SequenceStep("front_l", "8. FRONT-LEFT (30° TILT)", "ORBIT 315° & TILT 30° DOWN", 30, 315),
        SequenceStep("top", "9. TOP OVERHEAD (90° DOWN)", "HOLD OVERHEAD LOOKING DOWN", 90, 0),
        SequenceStep("bottom", "10. UNDERSIDE (60° UP)", "AIM UPWARD AT ASSET BASE", -60, 0)
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        rootLayout = findViewById(R.id.rootLayout)
        sceneView = findViewById(R.id.arSceneView)
        snapFlash = findViewById(R.id.snapFlash)

        targetBadge = findViewById(R.id.targetBadge)
        chkListLeft = findViewById(R.id.chkListLeft)
        chkListRight = findViewById(R.id.chkListRight)
        valDist = findViewById(R.id.valDist)
        valAlign = findViewById(R.id.valAlign)
        valScale = findViewById(R.id.valScale)

        distIndicator = findViewById(R.id.distIndicator)
        normalIndicator = findViewById(R.id.normalIndicator)
        centerIndicator = findViewById(R.id.centerIndicator)

        scaleBar = findViewById(R.id.scaleBar)
        scaleReadout = findViewById(R.id.scaleReadout)
        btnScaleDown = findViewById(R.id.btnScaleDown)
        btnScaleUp = findViewById(R.id.btnScaleUp)
        forceSnapBtn = findViewById(R.id.forceSnapBtn)
        batchUploadBtn = findViewById(R.id.batchUploadBtn)
        actionButton = findViewById(R.id.actionButton)

        previewScrollView = findViewById(R.id.previewScrollView)
        previewContainer = findViewById(R.id.previewContainer)

        fullPreviewLayout = findViewById(R.id.fullPreviewLayout)
        fullImageView = findViewById(R.id.fullImageView)
        fullPreviewLabel = findViewById(R.id.fullPreviewLabel)
        closePreviewBtn = findViewById(R.id.closePreviewBtn)

        // Lift status banner above the center aiming zone
        (distIndicator.parent as? View)?.let { hudBanner ->
            hudBanner.post {
                val liftPx = -resources.displayMetrics.heightPixels * 0.22f
                hudBanner.translationY = liftPx
                hudBanner.alpha = 0.88f
            }
        }

        try {
            toneGen = ToneGenerator(AudioManager.STREAM_NOTIFICATION, 100)
        } catch (e: Exception) {}

        initMagicTouchModel()

        overlayView = PrismOverlayView(this)
        rootLayout.addView(overlayView, 1)

        buildThumbnailStrip()

        sceneView.sessionConfiguration = { session, config ->
            config.focusMode = Config.FocusMode.AUTO
            config.planeFindingMode = Config.PlaneFindingMode.HORIZONTAL
            config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                config.depthMode = Config.DepthMode.AUTOMATIC
            }
        }

        sceneView.planeRenderer.isEnabled = false
        sceneView.planeRenderer.isVisible = false

        sceneView.onSessionUpdated = { _, frame ->
            lastFrame = frame
            onTrackingFrame(frame)
        }

        actionButton.setOnClickListener {
            if (!isBoxPlaced) placePrismAtHitTest()
        }

        forceSnapBtn.setOnClickListener {
            if (!isCapturing && currentStepIdx < sequence.size) {
                triggerLocalCapture()
            }
        }

        batchUploadBtn.setOnClickListener {
            if (!isBatchUploading && capturedThumbnails.isNotEmpty()) {
                startBatchUpload()
            }
        }

        closePreviewBtn.setOnClickListener {
            fullPreviewLayout.visibility = View.GONE
        }

        btnScaleUp.setOnClickListener { scalePrism(1.08f) }
        btnScaleDown.setOnClickListener { scalePrism(0.92f) }

        scaleReadout.setOnClickListener {
            initialAzimuth += Math.toRadians(15.0).toFloat()
            requestOverlayRedraw()
            Toast.makeText(this, "Rotated +15° (Long-press for 90°)", Toast.LENGTH_SHORT).show()
        }
        valScale.setOnClickListener { showDebugDialog() }
        targetBadge.setOnClickListener { showDebugDialog() }

        scaleReadout.setOnLongClickListener {
            val temp = prismWidth
            prismWidth = prismDepth
            prismDepth = temp
            initialAzimuth += (Math.PI / 2.0).toFloat()
            updateScaleLabels()
            requestOverlayRedraw()
            Toast.makeText(this, "Rotated 90°", Toast.LENGTH_SHORT).show()
            true
        }

        updateChecklistUI()
        updateScaleLabels()
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            preLockAnchor?.detach()
            preLockAnchor = null
            prismAnchor?.detach()
            prismAnchor = null
            mlExecutor.execute {
                try { magicTouchInterpreter?.close() } catch (_: Exception) {}
                magicTouchInterpreter = null
            }
            mlExecutor.shutdown()
        } catch (e: Exception) {}
    }

    private fun requestOverlayRedraw() {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            overlayView.invalidate()
        } else {
            overlayView.postInvalidate()
        }
    }

    private fun setTextIfChanged(tv: TextView, text: String, colorInt: Int? = null) {
        if (tv.text.toString() != text) {
            tv.text = text
        }
        if (colorInt != null && tv.currentTextColor != colorInt) {
            tv.setTextColor(colorInt)
        }
    }

    private fun logDebug(msg: String) {
        Log.d("RigDebug", msg)
        debugLog.append("[${System.currentTimeMillis() % 100000}] $msg\n")
    }

    private fun showDebugDialog() {
        runOnUiThread {
            val tv = TextView(this).apply {
                text = debugLog.toString()
                textSize = 12f
                setTextColor(Color.WHITE)
                setBackgroundColor(Color.parseColor("#0D1117"))
                setPadding(32, 32, 32, 32)
                setTextIsSelectable(true)
            }
            val sv = ScrollView(this).apply { addView(tv) }
            AlertDialog.Builder(this)
                .setTitle("Auto-Snap MagicTouch Diagnostics")
                .setView(sv)
                .setPositiveButton("OK", null)
                .setNeutralButton("Clear") { _, _ -> debugLog.clear() }
                .show()
        }
    }

    private fun initMagicTouchModel() {
        mlExecutor.execute {
            try {
                logDebug("Loading $magicTouchAssetName on CPU-XNNPACK...")
                val fd: AssetFileDescriptor = assets.openFd(magicTouchAssetName)
                val inputStream = FileInputStream(fd.fileDescriptor)
                val fileChannel = inputStream.channel
                val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)

                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseXNNPACK(true)
                }
                val interp = Interpreter(modelBuffer, options)

                val inTensor = interp.getInputTensor(0)
                val inShape = inTensor.shape()
                mtInH = if (inShape.size >= 3) inShape[1] else 512
                mtInW = if (inShape.size >= 3) inShape[2] else 512
                mtInC = if (inShape.size >= 4) inShape[3] else 4
                mtInType = inTensor.dataType()

                val outTensor = interp.getOutputTensor(0)
                mtOutShape = outTensor.shape()
                mtOutType = outTensor.dataType()

                mtInputBuf = ByteBuffer.allocateDirect(inTensor.numBytes()).order(ByteOrder.nativeOrder())
                mtOutputBuf = ByteBuffer.allocateDirect(outTensor.numBytes()).order(ByteOrder.nativeOrder())
                mtPixelBuf = IntArray(mtInW * mtInH)

                if (mtInType == DataType.FLOAT32) {
                    mtFloatInput = FloatArray(mtInW * mtInH * mtInC)
                }
                if (mtOutType == DataType.FLOAT32) {
                    var outElems = 1
                    for (d in mtOutShape) outElems *= d
                    mtFloatOutput = FloatArray(outElems)
                }

                // Pre-compute center (0.5, 0.5) Gaussian prompt map once to eliminate 262k exp() calls per snap
                val gMap = FloatArray(mtInW * mtInH)
                val cx = mtInW * 0.5f
                val cy = mtInH * 0.5f
                val invTwoSigmaSq = 1.0f / (2.0f * 12.0f * 12.0f)
                var idx = 0
                for (y in 0 until mtInH) {
                    val dy = y - cy
                    val dySq = dy * dy
                    for (x in 0 until mtInW) {
                        val dx = x - cx
                        gMap[idx++] = exp(-(dx * dx + dySq) * invTwoSigmaSq)
                    }
                }
                centerGaussMap = gMap

                magicTouchInterpreter = interp
                logDebug("✓ MagicTouch Ready [CPU-XNNPACK]! In:${inShape.contentToString()} Out:${mtOutShape.contentToString()}")
            } catch (e: Exception) {
                logDebug("❌ MagicTouch Load Failed: ${e.javaClass.simpleName}: ${e.message}")
            }
        }
    }

    private fun buildThumbnailStrip() {
        previewContainer.removeAllViews()
        previewCardViews.clear()

        val dpWidth = (64 * resources.displayMetrics.density).toInt()
        val dpHeight = (64 * resources.displayMetrics.density).toInt()
        val dpMargin = (4 * resources.displayMetrics.density).toInt()

        for (i in sequence.indices) {
            val frame = FrameLayout(this)
            val params = LinearLayout.LayoutParams(dpWidth, dpHeight).apply {
                setMargins(dpMargin, 0, dpMargin, 0)
            }
            frame.layoutParams = params
            frame.setBackgroundColor(Color.parseColor("#21262D"))

            val iv = ImageView(this).apply {
                layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
                scaleType = ImageView.ScaleType.CENTER_CROP
            }

            val badge = TextView(this).apply {
                layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT).apply {
                    gravity = android.view.Gravity.BOTTOM
                }
                text = "${i + 1}"
                textSize = 9f
                setTextColor(Color.WHITE)
                setBackgroundColor(Color.parseColor("#B30D1117"))
                gravity = android.view.Gravity.CENTER
            }

            frame.addView(iv)
            frame.addView(badge)
            frame.setOnClickListener { showFullPreview(i) }

            previewContainer.addView(frame)
            previewCardViews.add(iv)
        }
    }

    private fun showFullPreview(idx: Int) {
        val file = File(cacheDir, "plate_${sequence[idx].id}.jpg")
        if (file.exists()) {
            val bmp = BitmapFactory.decodeFile(file.absolutePath)
            fullImageView.setImageBitmap(bmp)
            fullPreviewLabel.text = "${sequence[idx].label} (LOCAL FILE SAVED)"
            fullPreviewLayout.visibility = View.VISIBLE
        } else {
            Toast.makeText(this, "Facet ${idx + 1} not captured yet", Toast.LENGTH_SHORT).show()
        }
    }

    private fun scalePrism(factor: Float) {
        prismWidth = (prismWidth * factor).coerceIn(0.04f, 0.50f)
        prismDepth = (prismDepth * factor).coerceIn(0.04f, 0.50f)
        prismHeight = (prismHeight * factor).coerceIn(0.02f, 0.40f)
        updateScaleLabels()
        requestOverlayRedraw()
    }

    private fun updateScaleLabels() {
        val wCm = (prismWidth * 100).roundToInt()
        val dCm = (prismDepth * 100).roundToInt()
        val hCm = (prismHeight * 100).roundToInt()
        scaleReadout.text = "PRISM: ${wCm}×${dCm}×${hCm}cm"
        valScale.text = "SCALE: ${wCm}×${dCm}"
    }

    private fun updateChecklistUI() {
        val sbL = StringBuilder()
        val sbR = StringBuilder()
        for (i in 0 until 5) {
            val icon = if (capturedThumbnails.containsKey(i)) "🟩" else if (i == currentStepIdx) "▶" else "⬜"
            sbL.append("$icon ${sequence[i].label.substring(3)}\n")
        }
        for (i in 5 until 10) {
            val icon = if (capturedThumbnails.containsKey(i)) "🟩" else if (i == currentStepIdx) "▶" else "⬜"
            sbR.append("$icon ${sequence[i].label.substring(3)}\n")
        }
        chkListLeft.text = sbL.toString().trimEnd()
        chkListRight.text = sbR.toString().trimEnd()

        val capturedCount = capturedThumbnails.size
        batchUploadBtn.text = if (capturedCount == 10) "UPLOAD ALL 10 PLATES TO MODAL" else "UPLOAD PLATES TO MODAL ($capturedCount/10)"
        batchUploadBtn.visibility = if (capturedCount > 0) View.VISIBLE else View.GONE
    }

    // =========================================================================
    // STRICT 8-RAY TABLE-PLANE GROUNDING & ARCORE 16-BIT DEPTH HEIGHT
    // =========================================================================
    private fun followSubsumedPlane(plane: Plane): Plane {
        var p = plane
        while (p.subsumedBy != null) {
            p = p.subsumedBy!!
        }
        return p
    }

    private fun getActiveRootPlane(): Plane? {
        val p = trackedTablePlane ?: return null
        val root = followSubsumedPlane(p)
        if (root.trackingState == TrackingState.TRACKING) {
            trackedTablePlane = root
            return root
        }
        return null
    }

    private fun resolveStrictTableSurface(frame: Frame): Pair<Plane?, Pose?> {
        val sW = sceneView.width.toFloat()
        val sH = sceneView.height.toFloat()
        if (sW <= 0f || sH <= 0f) return Pair(null, null)

        val cx = sW * 0.5f
        val cy = sH * 0.5f
        val centerHits = frame.hitTest(cx, cy)

        val centerPlaneHit = centerHits.firstOrNull { hit ->
            val t = hit.trackable
            t is Plane &&
                t.trackingState == TrackingState.TRACKING &&
                t.type == Plane.Type.HORIZONTAL_UPWARD_FACING
        }
        if (centerPlaneHit != null) {
            val rootPlane = followSubsumedPlane(centerPlaneHit.trackable as Plane)
            trackedTablePlane = rootPlane
            val tableY = rootPlane.centerPose.ty()
            val pose = Pose(
                floatArrayOf(centerPlaneHit.hitPose.tx(), tableY, centerPlaneHit.hitPose.tz()),
                floatArrayOf(0f, 0f, 0f, 1f)
            )
            return Pair(rootPlane, pose)
        }

        // Center ray hit the elevated asset -> cast 8 perimeter rays on the surrounding table
        val minDim = min(sW, sH)
        val ringR = minDim * 0.28f
        val planeVotes = mutableMapOf<Plane, Int>()
        var fallbackPose: Pose? = null

        for (i in 0 until 8) {
            val ang = (2.0 * Math.PI * i / 8.0).toFloat()
            val px = (cx + ringR * cos(ang)).coerceIn(8f, sW - 9f)
            val py = (cy + ringR * sin(ang)).coerceIn(8f, sH - 9f)
            val hits = frame.hitTest(px, py)
            val pHit = hits.firstOrNull { hit ->
                val t = hit.trackable
                t is Plane &&
                    t.trackingState == TrackingState.TRACKING &&
                    t.type == Plane.Type.HORIZONTAL_UPWARD_FACING
            }
            if (pHit != null) {
                val rootP = followSubsumedPlane(pHit.trackable as Plane)
                planeVotes[rootP] = (planeVotes[rootP] ?: 0) + 1
                if (fallbackPose == null) fallbackPose = pHit.hitPose
            }
        }

        val winningPlane = planeVotes.maxByOrNull { it.value }?.key ?: getActiveRootPlane()
        if (winningPlane != null) {
            trackedTablePlane = winningPlane
            val tableY = winningPlane.centerPose.ty()
            val invVp = FloatArray(16)
            if (Matrix.invertM(invVp, 0, frameVpMatrix, 0)) {
                val centerOnTable = intersectScreenRayWithHorizontalPlane(cx, cy, sW, sH, invVp, tableY)
                if (centerOnTable != null) {
                    return Pair(winningPlane, Pose(centerOnTable, floatArrayOf(0f, 0f, 0f, 1f)))
                }
            }
            val ref = fallbackPose ?: winningPlane.centerPose
            return Pair(winningPlane, Pose(floatArrayOf(ref.tx(), tableY, ref.tz()), floatArrayOf(0f, 0f, 0f, 1f)))
        }

        return Pair(null, null)
    }

    private fun measureObjectHeightSync(
        frame: Frame,
        tableSurfaceY: Float,
        centerScreenX: Float,
        centerScreenY: Float,
        refWorldX: Float,
        refWorldZ: Float,
        radiusMeters: Float
    ): Float {
        val sW = sceneView.width.toFloat()
        val sH = sceneView.height.toFloat()
        if (sW <= 0f || sH <= 0f) return 0f

        val invVp = FloatArray(16)
        val hasInvVp = Matrix.invertM(invVp, 0, frameVpMatrix, 0)

        if (hasInvVp) {
            var depthImg: Image? = null
            try {
                depthImg = frame.acquireDepthImage16Bits()
                val dW = depthImg.width
                val dH = depthImg.height
                val plane = depthImg.planes[0]
                val buf = plane.buffer
                val rowStride = plane.rowStride
                val pixStride = plane.pixelStride

                val minDim = min(sW, sH)
                val sampleR = minDim * 0.14f
                val grid = 5
                val viewPts = FloatArray(grid * grid * 2)
                val texPts = FloatArray(grid * grid * 2)
                var idx = 0
                for (gy in 0 until grid) {
                    val fy = (gy / (grid - 1f)) * 2f - 1f
                    val sy = (centerScreenY + fy * sampleR).coerceIn(8f, sH - 9f)
                    for (gx in 0 until grid) {
                        val fx = (gx / (grid - 1f)) * 2f - 1f
                        val sx = (centerScreenX + fx * sampleR).coerceIn(8f, sW - 9f)
                        viewPts[idx++] = sx
                        viewPts[idx++] = sy
                    }
                }

                frame.transformCoordinates2d(
                    Coordinates2d.VIEW, viewPts,
                    Coordinates2d.TEXTURE_NORMALIZED, texPts
                )

                val camPose = frame.camera.pose
                val camFwdX = -camPose.zAxis[0]
                val camFwdY = -camPose.zAxis[1]
                val camFwdZ = -camPose.zAxis[2]

                val heights = mutableListOf<Float>()
                for (i in 0 until (grid * grid)) {
                    val dx = (texPts[i * 2] * dW).toInt().coerceIn(0, dW - 1)
                    val dy = (texPts[i * 2 + 1] * dH).toInt().coerceIn(0, dH - 1)
                    val offset = dy * rowStride + dx * pixStride
                    val lo = buf.get(offset).toInt() and 0xFF
                    val hi = buf.get(offset + 1).toInt() and 0xFF
                    val depthMm = lo or (hi shl 8)
                    if (depthMm !in 120..1200) continue

                    val depthM = depthMm * 0.001f
                    val sx = viewPts[i * 2]
                    val sy = viewPts[i * 2 + 1]
                    val ray = unprojectScreenRay(sx, sy, sW, sH, invVp) ?: continue
                    val cosTheta = ray[3] * camFwdX + ray[4] * camFwdY + ray[5] * camFwdZ
                    if (cosTheta <= 0.25f) continue

                    val rayDist = depthM / cosTheta
                    val wy = ray[1] + rayDist * ray[4]
                    val hAboveTable = wy - tableSurfaceY
                    if (hAboveTable in 0.010f..0.32f) {
                        heights.add(hAboveTable)
                    }
                }

                if (heights.size >= 4) {
                    heights.sort()
                    val p85 = heights[(heights.size * 0.85f).toInt().coerceAtMost(heights.lastIndex)]
                    return (p85 * 1.12f).coerceIn(0.028f, 0.28f)
                }
            } catch (_: Throwable) {
            } finally {
                try { depthImg?.close() } catch (_: Exception) {}
            }
        }

        try {
            val pc = frame.acquirePointCloud()
            val pBuf = pc.points
            val pCount = pBuf.remaining() / 4
            var maxY = 0f
            var validPts = 0
            val lim = radiusMeters.coerceIn(0.06f, 0.25f)
            for (p in 0 until pCount) {
                if (pBuf.get(p * 4 + 3) < 0.20f) continue
                val px = pBuf.get(p * 4) - refWorldX
                val py = pBuf.get(p * 4 + 1) - tableSurfaceY
                val pz = pBuf.get(p * 4 + 2) - refWorldZ
                if (py in 0.008f..0.30f && sqrt(px * px + pz * pz) <= lim) {
                    if (py > maxY) maxY = py
                    validPts++
                }
            }
            pc.close()
            if (validPts >= 3 && maxY > 0.012f) {
                return (maxY * 1.15f).coerceIn(0.028f, 0.25f)
            }
        } catch (_: Exception) {}

        return 0f
    }

    private fun createStabilizedTableAnchor(x: Float, y: Float, z: Float): Anchor? {
        val identityQuat = floatArrayOf(0f, 0f, 0f, 1f)
        val rootPlane = getActiveRootPlane()
        if (rootPlane != null) {
            try {
                val planePose = Pose(floatArrayOf(x, rootPlane.centerPose.ty(), z), identityQuat)
                return rootPlane.createAnchor(planePose)
            } catch (_: Exception) {}
        }
        return try {
            sceneView.session?.createAnchor(Pose(floatArrayOf(x, y, z), identityQuat))
        } catch (_: Exception) {
            null
        }
    }

    private fun resetAnchorDamper(pose: Pose) {
        stableAnchorX = pose.tx()
        stableAnchorY = pose.ty()
        stableAnchorZ = pose.tz()
        hasStableAnchor = true
    }

    private fun getStabilizedAnchorPose(rawPose: Pose): Pose {
        val rx = rawPose.tx()
        val ry = rawPose.ty()
        val rz = rawPose.tz()

        if (!hasStableAnchor) {
            stableAnchorX = rx; stableAnchorY = ry; stableAnchorZ = rz
            hasStableAnchor = true
        } else {
            val dx = rx - stableAnchorX
            val dy = ry - stableAnchorY
            val dz = rz - stableAnchorZ
            val dist = sqrt(dx * dx + dy * dy + dz * dz)

            when {
                dist < 0.0012f -> {}
                dist > 0.018f -> {
                    stableAnchorX = rx; stableAnchorY = ry; stableAnchorZ = rz
                }
                else -> {
                    val alpha = ((dist - 0.0012f) / 0.0168f).coerceIn(0.12f, 0.55f)
                    stableAnchorX += alpha * dx
                    stableAnchorY += alpha * dy
                    stableAnchorZ += alpha * dz
                }
            }
        }
        return Pose(floatArrayOf(stableAnchorX, stableAnchorY, stableAnchorZ), floatArrayOf(0f, 0f, 0f, 1f))
    }

    private fun releasePreLockPin() {
        hasLiveCling = false
        try { preLockAnchor?.detach() } catch (_: Exception) {}
        preLockAnchor = null
        hasDwellRef = false
        dwellProgress = 0f
    }

    // =========================================================================
    // PHASE 1: PAUSE-TRIGGERED AUTO-SNAP MAGICTOUCH (WORLD-PINNED AT 60 FPS)
    // =========================================================================
    private data class MagicTouch3DFit(
        val worldX: Float,
        val worldZ: Float,
        val widthM: Float,
        val depthM: Float,
        val headingRad: Float,
        val screenCX: Float,
        val screenCY: Float
    )

    private fun triggerAutoSnapMagicTouch(frame: Frame, tablePose: Pose) {
        val sW = sceneView.width
        val sH = sceneView.height
        if (sW <= 0 || sH <= 0 || isAutoSnapRunning || isBoxPlaced) return
        if (magicTouchInterpreter == null) return

        isAutoSnapRunning = true

        val snapVp = frameVpMatrix.copyOf()
        val snapInvVp = FloatArray(16)
        if (!Matrix.invertM(snapInvVp, 0, snapVp, 0)) {
            isAutoSnapRunning = false
            return
        }

        val tableY = tablePose.ty()
        val aimWorldX = tablePose.tx()
        val aimWorldZ = tablePose.tz()
        val camPose = frame.camera.pose
        val snapCamX = camPose.tx()
        val snapCamY = camPose.ty()
        val snapCamZ = camPose.tz()
        val snapFwdX = -camPose.zAxis[0]
        val snapFwdY = -camPose.zAxis[1]
        val snapFwdZ = -camPose.zAxis[2]

        val measuredHeightM = measureObjectHeightSync(
            frame, tableY, sW * 0.5f, sH * 0.5f, aimWorldX, aimWorldZ, 0.14f
        )

        // Downscale PixelCopy directly to 512 on the short axis (avoids full-res copy + rescale)
        val targetW: Int
        val targetH: Int
        if (sW <= sH) {
            targetW = mtInW
            targetH = max(mtInH, (mtInW.toFloat() * sH / sW).roundToInt())
        } else {
            targetH = mtInH
            targetW = max(mtInW, (mtInH.toFloat() * sW / sH).roundToInt())
        }
        if (preLockCopyBmp == null || preLockCopyBmp!!.width != targetW || preLockCopyBmp!!.height != targetH) {
            preLockCopyBmp = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        }
        val destBmp = preLockCopyBmp!!

        PixelCopy.request(
            sceneView as SurfaceView,
            destBmp,
            PixelCopy.OnPixelCopyFinishedListener { copyResult ->
                if (copyResult != PixelCopy.SUCCESS || isBoxPlaced) {
                    isAutoSnapRunning = false
                    return@OnPixelCopyFinishedListener
                }

                val bmpCopy = destBmp.copy(Bitmap.Config.ARGB_8888, false)
                mlExecutor.execute {
                    var squareSmall: Bitmap? = null
                    try {
                        val interp = magicTouchInterpreter ?: return@execute
                        if (isBoxPlaced) return@execute

                        val t0 = System.currentTimeMillis()
                        val minDimSmall = min(bmpCopy.width, bmpCopy.height)
                        val cropXSmall = (bmpCopy.width - minDimSmall) / 2
                        val cropYSmall = (bmpCopy.height - minDimSmall) / 2
                        squareSmall = Bitmap.createBitmap(bmpCopy, cropXSmall, cropYSmall, minDimSmall, minDimSmall)

                        val minDimFull = min(sW, sH).toFloat()
                        val cropXFull = (sW - minDimFull) * 0.5f
                        val cropYFull = (sH - minDimFull) * 0.5f
                        val estHeight = if (measuredHeightM > 0.015f) measuredHeightM else 0.045f

                        val fit = runMagicTouchSolveOnWorker(
                            interp = interp,
                            squareBmp = squareSmall,
                            normKeyX = 0.5f,
                            normKeyY = 0.5f,
                            cropXFull = cropXFull,
                            cropYFull = cropYFull,
                            minDimFull = minDimFull,
                            sWidth = sW.toFloat(),
                            sHeight = sH.toFloat(),
                            invVpMatrix = snapInvVp,
                            tableY = tableY,
                            estHeightM = estHeight,
                            camX = snapCamX,
                            camZ = snapCamZ
                        )
                        lastSnapLatencyMs = System.currentTimeMillis() - t0

                        // Discard if user whipped the camera away while snap was computing
                        val rotDot = snapFwdX * liveCamFwdX + snapFwdY * liveCamFwdY + snapFwdZ * liveCamFwdZ
                        val camShift = sqrt(
                            (liveCamX - snapCamX).pow(2) +
                                (liveCamY - snapCamY).pow(2) +
                                (liveCamZ - snapCamZ).pow(2)
                        )
                        if (rotDot < 0.94f || camShift > 0.10f) {
                            return@execute
                        }

                        runOnUiThread {
                            if (isBoxPlaced) return@runOnUiThread

                            if (fit != null) {
                                val targetH = if (measuredHeightM > 0.015f) {
                                    measuredHeightM
                                } else {
                                    (min(fit.widthM, fit.depthM) * 0.45f).coerceIn(0.03f, 0.14f)
                                }

                                // Pin the snapped object to a real ARCore table-plane Anchor (60 FPS 0-lag tracking!)
                                val newPin = createStabilizedTableAnchor(fit.worldX, tableY, fit.worldZ)
                                try { preLockAnchor?.detach() } catch (_: Exception) {}
                                preLockAnchor = newPin

                                if (newPin != null) {
                                    val p = newPin.pose
                                    liveWorldX = p.tx()
                                    liveWorldY = p.ty()
                                    liveWorldZ = p.tz()
                                    resetAnchorDamper(p)
                                } else {
                                    liveWorldX = fit.worldX
                                    liveWorldY = tableY
                                    liveWorldZ = fit.worldZ
                                }

                                liveWidth = fit.widthM
                                liveDepth = fit.depthM
                                liveHeight = targetH
                                liveAzimuth = fit.headingRad
                                hasLiveCling = true
                                hasFailedSpot = false
                                dwellProgress = 1f

                                logDebug("✓ Auto-Snapped (${lastSnapLatencyMs}ms): ${(liveWidth * 100).roundToInt()}×${(liveDepth * 100).roundToInt()}×${(liveHeight * 100).roundToInt()}cm")
                                try {
                                    toneGen?.startTone(ToneGenerator.TONE_PROP_BEEP2, 60)
                                } catch (_: Exception) {}
                                requestOverlayRedraw()
                            } else {
                                // Empty table spot -> record cooldown so we don't re-fire until user moves aim
                                hasFailedSpot = true
                                failedSpotX = aimWorldX
                                failedSpotZ = aimWorldZ
                                failedSpotTimeMs = System.currentTimeMillis()
                                hasDwellRef = false
                                dwellProgress = 0f
                            }
                        }
                    } catch (e: Exception) {
                        logDebug("⚠️ Auto-Snap Err: ${e.message}")
                    } finally {
                        try { squareSmall?.recycle() } catch (_: Exception) {}
                        try { bmpCopy.recycle() } catch (_: Exception) {}
                        isAutoSnapRunning = false
                    }
                }
            },
            mainHandler
        )
    }

    // =========================================================================
    // PHASE 2: ZERO-JUMP LOCK HANDOFF (OR FALLBACK ONE-SHOT MAGICTOUCH)
    // =========================================================================
    private fun placePrismAtHitTest() {
        val frame = lastFrame ?: return
        val (tablePlane, tablePose) = resolveStrictTableSurface(frame)

        if (tablePlane == null || tablePose == null) {
            Toast.makeText(this, "Move phone slightly so table surface plane locks first", Toast.LENGTH_SHORT).show()
            return
        }

        val tableY = tablePlane.centerPose.ty()
        val usedAutoSnap = hasLiveCling
        val anchor: Anchor?

        if (usedAutoSnap) {
            // Promote the already-pinned MagicTouch anchor directly (0ms instant WYSIWYG lock!)
            val existingPin = preLockAnchor
            if (existingPin != null && existingPin.trackingState == TrackingState.TRACKING) {
                anchor = existingPin
                preLockAnchor = null
            } else {
                try { preLockAnchor?.detach() } catch (_: Exception) {}
                preLockAnchor = null
                anchor = createStabilizedTableAnchor(liveWorldX, tableY, liveWorldZ)
            }
            prismWidth = liveWidth
            prismDepth = liveDepth
            prismHeight = liveHeight
            initialAzimuth = liveAzimuth
        } else {
            try { preLockAnchor?.detach() } catch (_: Exception) {}
            preLockAnchor = null
            anchor = createStabilizedTableAnchor(tablePose.tx(), tableY, tablePose.tz())
            val camPose = frame.camera.pose
            initialAzimuth = atan2(camPose.tx() - tablePose.tx(), camPose.tz() - tablePose.tz())
        }

        if (anchor == null) {
            Toast.makeText(this, "Table anchor not ready — move slightly", Toast.LENGTH_SHORT).show()
            return
        }

        prismAnchor?.detach()
        prismAnchor = anchor
        resetAnchorDamper(anchor.pose)
        isBoxPlaced = true

        val vMat = FloatArray(16)
        val pMat = FloatArray(16)
        val vpMat = FloatArray(16)
        val invVpMat = FloatArray(16)

        frame.camera.getViewMatrix(vMat, 0)
        frame.camera.getProjectionMatrix(pMat, 0, 0.05f, 50.0f)
        Matrix.multiplyMM(vpMat, 0, pMat, 0, vMat, 0)
        Matrix.invertM(invVpMat, 0, vpMat, 0)

        val snapshotAnchorPose = Pose(
            floatArrayOf(anchor.pose.tx(), tableY, anchor.pose.tz()),
            floatArrayOf(0f, 0f, 0f, 1f)
        )
        val camPose = frame.camera.pose
        overlayView.updateLockedSnapshot(vpMat, snapshotAnchorPose, camPose)

        if (usedAutoSnap) {
            val wCm = (prismWidth * 100).roundToInt()
            val dCm = (prismDepth * 100).roundToInt()
            val hCm = (prismHeight * 100).roundToInt()
            logDebug("✓ Instant WYSIWYG Lock: ${wCm}×${dCm}×${hCm}cm")
            Toast.makeText(this, "✓ Locked: ${wCm}×${dCm}×${hCm}cm", Toast.LENGTH_SHORT).show()
        } else {
            // User tapped Lock before auto-snap fired -> run one-shot MagicTouch immediately
            val lockHeightM = measureObjectHeightSync(
                frame, tableY, sceneView.width * 0.5f, sceneView.height * 0.5f,
                snapshotAnchorPose.tx(), snapshotAnchorPose.tz(), 0.14f
            )
            if (lockHeightM > 0.015f) prismHeight = lockHeightM

            runFallbackLockMagicTouch(
                invVpMat, tableY, camPose.tx(), camPose.tz(), prismHeight
            )
        }

        actionButton.visibility = View.GONE
        scaleBar.visibility = View.VISIBLE
        previewScrollView.visibility = View.VISIBLE
        forceSnapBtn.visibility = View.VISIBLE
        forceSnapBtn.text = "FORCE SNAP [${sequence[currentStepIdx].id.uppercase(Locale.US)}]"
        targetBadge.text = sequence[currentStepIdx].label
        updateChecklistUI()
        updateScaleLabels()
        requestOverlayRedraw()

        try {
            toneGen?.startTone(ToneGenerator.TONE_PROP_ACK, 140)
        } catch (e: Exception) {}
    }

    private fun runFallbackLockMagicTouch(
        invVpMatrix: FloatArray,
        tableY: Float,
        camX: Float,
        camZ: Float,
        seedHeightM: Float
    ) {
        val sWidth = sceneView.width
        val sHeight = sceneView.height
        if (sWidth <= 0 || sHeight <= 0) return

        val bitmap = Bitmap.createBitmap(sWidth, sHeight, Bitmap.Config.ARGB_8888)
        PixelCopy.request(
            sceneView as SurfaceView,
            bitmap,
            PixelCopy.OnPixelCopyFinishedListener { copyResult ->
                if (copyResult == PixelCopy.SUCCESS) {
                    mlExecutor.execute {
                        var squareBmp: Bitmap? = null
                        try {
                            val interp = magicTouchInterpreter ?: return@execute
                            val t0 = System.currentTimeMillis()

                            val minDim = min(bitmap.width, bitmap.height)
                            val cropX = (bitmap.width - minDim) / 2f
                            val cropY = (bitmap.height - minDim) / 2f
                            squareBmp = Bitmap.createBitmap(bitmap, cropX.toInt(), cropY.toInt(), minDim, minDim)

                            val fit = runMagicTouchSolveOnWorker(
                                interp = interp,
                                squareBmp = squareBmp,
                                normKeyX = 0.5f,
                                normKeyY = 0.5f,
                                cropXFull = cropX,
                                cropYFull = cropY,
                                minDimFull = minDim.toFloat(),
                                sWidth = sWidth.toFloat(),
                                sHeight = sHeight.toFloat(),
                                invVpMatrix = invVpMatrix,
                                tableY = tableY,
                                estHeightM = seedHeightM,
                                camX = camX,
                                camZ = camZ
                            )

                            val elapsed = System.currentTimeMillis() - t0
                            if (fit != null) {
                                val lockedFit = fit
                                runOnUiThread {
                                    try {
                                        val refinedAnchor = createStabilizedTableAnchor(lockedFit.worldX, tableY, lockedFit.worldZ)
                                        if (refinedAnchor != null) {
                                            prismAnchor?.detach()
                                            prismAnchor = refinedAnchor
                                            resetAnchorDamper(refinedAnchor.pose)
                                        }
                                    } catch (e: Exception) {
                                        logDebug("Re-anchor failed: ${e.message}")
                                    }

                                    initialAzimuth = lockedFit.headingRad
                                    prismWidth = lockedFit.widthM
                                    prismDepth = lockedFit.depthM
                                    updateScaleLabels()
                                    requestOverlayRedraw()

                                    val wCm = (prismWidth * 100).roundToInt()
                                    val dCm = (prismDepth * 100).roundToInt()
                                    val hCm = (prismHeight * 100).roundToInt()
                                    logDebug("✓ MagicTouch Lock (${elapsed}ms): ${wCm}×${dCm}×${hCm}cm")
                                    Toast.makeText(
                                        this@MainActivity,
                                        "✓ MagicTouch Locked: ${wCm}×${dCm}×${hCm}cm",
                                        Toast.LENGTH_SHORT
                                    ).show()
                                }
                            }
                        } catch (e: Exception) {
                            logDebug("❌ MagicTouch Lock Err: ${e.javaClass.simpleName}: ${e.message}")
                        } finally {
                            try { squareBmp?.recycle() } catch (_: Exception) {}
                            try { bitmap.recycle() } catch (_: Exception) {}
                        }
                    }
                }
            },
            mainHandler
        )
    }

    private fun runMagicTouchSolveOnWorker(
        interp: Interpreter,
        squareBmp: Bitmap,
        normKeyX: Float,
        normKeyY: Float,
        cropXFull: Float,
        cropYFull: Float,
        minDimFull: Float,
        sWidth: Float,
        sHeight: Float,
        invVpMatrix: FloatArray,
        tableY: Float,
        estHeightM: Float,
        camX: Float,
        camZ: Float
    ): MagicTouch3DFit? {
        val scaled = if (squareBmp.width == mtInW && squareBmp.height == mtInH) {
            squareBmp
        } else {
            Bitmap.createScaledBitmap(squareBmp, mtInW, mtInH, true)
        }

        val pixCount = mtInW * mtInH
        val pixels = mtPixelBuf ?: IntArray(pixCount).also { mtPixelBuf = it }
        scaled.getPixels(pixels, 0, mtInW, 0, 0, mtInW, mtInH)
        if (scaled !== squareBmp) scaled.recycle()

        val inputBuf = mtInputBuf ?: return null
        val outputBuf = mtOutputBuf ?: return null
        inputBuf.clear()

        val useCenterGauss = abs(normKeyX - 0.5f) < 0.002f && abs(normKeyY - 0.5f) < 0.002f && centerGaussMap != null
        val keyPxX = normKeyX * mtInW
        val keyPxY = normKeyY * mtInH
        val invTwoSigmaSq = 1.0f / (2.0f * 12.0f * 12.0f)
        val inv255 = 1.0f / 255.0f

        if (mtInType == DataType.FLOAT32) {
            val fIn = mtFloatInput ?: FloatArray(pixCount * mtInC).also { mtFloatInput = it }
            var outIdx = 0
            if (useCenterGauss) {
                val gMap = centerGaussMap!!
                if (mtInC >= 4) {
                    for (i in 0 until pixCount) {
                        val px = pixels[i]
                        fIn[outIdx++] = ((px shr 16) and 0xFF) * inv255
                        fIn[outIdx++] = ((px shr 8) and 0xFF) * inv255
                        fIn[outIdx++] = (px and 0xFF) * inv255
                        fIn[outIdx++] = gMap[i]
                    }
                } else {
                    for (i in 0 until pixCount) {
                        val px = pixels[i]
                        fIn[outIdx++] = ((px shr 16) and 0xFF) * inv255
                        fIn[outIdx++] = ((px shr 8) and 0xFF) * inv255
                        fIn[outIdx++] = (px and 0xFF) * inv255
                    }
                }
            } else {
                var pIdx = 0
                for (y in 0 until mtInH) {
                    val dy = y - keyPxY
                    val dySq = dy * dy
                    for (x in 0 until mtInW) {
                        val px = pixels[pIdx++]
                        fIn[outIdx++] = ((px shr 16) and 0xFF) * inv255
                        fIn[outIdx++] = ((px shr 8) and 0xFF) * inv255
                        fIn[outIdx++] = (px and 0xFF) * inv255
                        if (mtInC >= 4) {
                            val dx = x - keyPxX
                            fIn[outIdx++] = exp(-(dx * dx + dySq) * invTwoSigmaSq)
                        }
                    }
                }
            }
            inputBuf.asFloatBuffer().put(fIn)
        } else {
            var pIdx = 0
            val gMap = if (useCenterGauss) centerGaussMap else null
            for (y in 0 until mtInH) {
                val dy = y - keyPxY
                val dySq = dy * dy
                for (x in 0 until mtInW) {
                    val px = pixels[pIdx]
                    inputBuf.put(((px shr 16) and 0xFF).toByte())
                    inputBuf.put(((px shr 8) and 0xFF).toByte())
                    inputBuf.put((px and 0xFF).toByte())
                    if (mtInC >= 4) {
                        val g = gMap?.get(pIdx) ?: exp(-((x - keyPxX) * (x - keyPxX) + dySq) * invTwoSigmaSq)
                        inputBuf.put((g * 255f).toInt().coerceIn(0, 255).toByte())
                    }
                    pIdx++
                }
            }
        }
        inputBuf.rewind()
        outputBuf.clear()

        interp.run(inputBuf, outputBuf)
        outputBuf.rewind()

        val mH: Int; val mW: Int; val mC: Int
        when (mtOutShape.size) {
            4 -> if (mtOutShape[3] <= 4) { mH = mtOutShape[1]; mW = mtOutShape[2]; mC = mtOutShape[3] } else { mC = mtOutShape[1]; mH = mtOutShape[2]; mW = mtOutShape[3] }
            3 -> { mH = mtOutShape[1]; mW = mtOutShape[2]; mC = 1 }
            else -> return null
        }

        val floatOut = if (mtOutType == DataType.FLOAT32) {
            val arr = mtFloatOutput ?: FloatArray(mW * mH * mC).also { mtFloatOutput = it }
            outputBuf.asFloatBuffer().get(arr)
            arr
        } else {
            null
        }

        fun scoreAt(mx: Int, my: Int): Float {
            val i = my * mW + mx
            return if (floatOut != null) {
                if (mC == 1) floatOut[i] else (floatOut[i * mC + (mC - 1)] - floatOut[i * mC])
            } else {
                (outputBuf.get(i * mC + (mC - 1)).toInt() and 0xFF) * inv255
            }
        }

        val seedX = (normKeyX * mW).toInt().coerceIn(8, mW - 9)
        val seedY = (normKeyY * mH).toInt().coerceIn(8, mH - 9)
        var seedSum = 0f; var seedCnt = 0
        for (dy in -4..4) {
            for (dx in -4..4) {
                seedSum += scoreAt(seedX + dx, seedY + dy)
                seedCnt++
            }
        }
        val seedMean = seedSum / seedCnt

        var borderSum = 0f; var borderCnt = 0
        for (x in 0 until mW step 4) {
            borderSum += scoreAt(x, 2) + scoreAt(x, mH - 3)
            borderCnt += 2
        }
        for (y in 0 until mH step 4) {
            borderSum += scoreAt(2, y) + scoreAt(mW - 3, y)
            borderCnt += 2
        }
        val borderMean = borderSum / borderCnt.coerceAtLeast(1)

        if (abs(seedMean - borderMean) < 0.04f) return null

        val thresh = borderMean + (seedMean - borderMean) * 0.48f
        val seedIsHigher = seedMean > borderMean

        val gSize = 128
        val gridMask = BooleanArray(gSize * gSize)
        for (gy in 1 until gSize - 1) {
            val my = (gy * mH) / gSize
            for (gx in 1 until gSize - 1) {
                val mx = (gx * mW) / gSize
                val s = scoreAt(mx, my)
                gridMask[gy * gSize + gx] = if (seedIsHigher) (s >= thresh) else (s <= thresh)
            }
        }

        val seedGX = (normKeyX * gSize).toInt().coerceIn(4, gSize - 5)
        val seedGY = (normKeyY * gSize).toInt().coerceIn(4, gSize - 5)
        var startIdx = -1
        var bestDist = Int.MAX_VALUE
        for (gy in (seedGY - 8)..(seedGY + 8)) {
            for (gx in (seedGX - 8)..(seedGX + 8)) {
                if (gx in 1 until (gSize - 1) && gy in 1 until (gSize - 1)) {
                    val idx = gy * gSize + gx
                    if (gridMask[idx]) {
                        val d = (gx - seedGX) * (gx - seedGX) + (gy - seedGY) * (gy - seedGY)
                        if (d < bestDist) { bestDist = d; startIdx = idx }
                    }
                }
            }
        }
        if (startIdx == -1) return null

        val visited = BooleanArray(gSize * gSize)
        val queue = ArrayDeque<Int>()
        queue.add(startIdx)
        visited[startIdx] = true

        val fgScreenX = mutableListOf<Float>()
        val fgScreenY = mutableListOf<Float>()
        var sumX = 0f; var sumY = 0f
        var touchesLeft = false; var touchesRight = false
        var touchesTop = false; var touchesBottom = false

        while (!queue.isEmpty()) {
            val curr = queue.removeFirst()
            val cx = curr % gSize
            val cy = curr / gSize

            if (cx <= 2) touchesLeft = true
            if (cx >= gSize - 3) touchesRight = true
            if (cy <= 2) touchesTop = true
            if (cy >= gSize - 3) touchesBottom = true

            val sx = cropXFull + ((cx + 0.5f) / gSize) * minDimFull
            val sy = cropYFull + ((cy + 0.5f) / gSize) * minDimFull
            fgScreenX.add(sx); fgScreenY.add(sy)
            sumX += sx; sumY += sy

            for (ny in (cy - 1)..(cy + 1)) {
                for (nx in (cx - 1)..(cx + 1)) {
                    if (nx in 1 until (gSize - 1) && ny in 1 until (gSize - 1)) {
                        val nIdx = ny * gSize + nx
                        if (gridMask[nIdx] && !visited[nIdx]) {
                            visited[nIdx] = true
                            queue.add(nIdx)
                        }
                    }
                }
            }
        }

        val count = fgScreenX.size
        val edgeCount = (if (touchesLeft) 1 else 0) + (if (touchesRight) 1 else 0) +
            (if (touchesTop) 1 else 0) + (if (touchesBottom) 1 else 0)
        if (count < 25 || count > (gSize * gSize * 0.65f).toInt() || edgeCount >= 3) {
            return null
        }

        val barX = sumX / count
        val barY = sumY / count

        // Full 1° Rotating-Calipers minimum-area bounding box
        var bestArea = Float.MAX_VALUE
        var bestRad = 0f
        for (deg in 0 until 90) {
            val rad = Math.toRadians(deg.toDouble()).toFloat()
            val c = cos(rad); val s = sin(rad)
            var minU = Float.MAX_VALUE; var maxU = -Float.MAX_VALUE
            var minV = Float.MAX_VALUE; var maxV = -Float.MAX_VALUE

            for (i in 0 until count) {
                val dx = fgScreenX[i] - barX
                val dy = fgScreenY[i] - barY
                val pu = dx * c + dy * s
                val pv = -dx * s + dy * c
                if (pu < minU) minU = pu
                if (pu > maxU) maxU = pu
                if (pv < minV) minV = pv
                if (pv > maxV) maxV = pv
            }
            val area = (maxU - minU).coerceAtLeast(10f) * (maxV - minV).coerceAtLeast(10f)
            if (area < bestArea) { bestArea = area; bestRad = rad }
        }

        val uX = cos(bestRad); val uY = sin(bestRad)
        val vX = -uY; val vY = uX

        var minU = Float.MAX_VALUE; var maxU = -Float.MAX_VALUE
        var minV = Float.MAX_VALUE; var maxV = -Float.MAX_VALUE
        for (i in 0 until count) {
            val dx = fgScreenX[i] - barX
            val dy = fgScreenY[i] - barY
            val pu = dx * uX + dy * uY
            val pv = dx * vX + dy * vY
            if (pu < minU) minU = pu
            if (pu > maxU) maxU = pu
            if (pv < minV) minV = pv
            if (pv > maxV) maxV = pv
        }

        val midU = (minU + maxU) * 0.5f
        val midV = (minV + maxV) * 0.5f
        val halfSpan1Px = ((maxU - minU) * 0.5f).coerceAtLeast(14f)
        val halfSpan2Px = ((maxV - minV) * 0.5f).coerceAtLeast(14f)

        val centerScreenX = barX + midU * uX + midV * vX
        val centerScreenY = barY + midU * uY + midV * vY

        val midPlaneY = tableY + estHeightM * 0.5f

        val pCenterMid = intersectScreenRayWithHorizontalPlane(centerScreenX, centerScreenY, sWidth, sHeight, invVpMatrix, midPlaneY)
        val p1AMid = intersectScreenRayWithHorizontalPlane(centerScreenX + halfSpan1Px * uX, centerScreenY + halfSpan1Px * uY, sWidth, sHeight, invVpMatrix, midPlaneY)
        val p1BMid = intersectScreenRayWithHorizontalPlane(centerScreenX - halfSpan1Px * uX, centerScreenY - halfSpan1Px * uY, sWidth, sHeight, invVpMatrix, midPlaneY)
        val p2AMid = intersectScreenRayWithHorizontalPlane(centerScreenX + halfSpan2Px * vX, centerScreenY + halfSpan2Px * vY, sWidth, sHeight, invVpMatrix, midPlaneY)
        val p2BMid = intersectScreenRayWithHorizontalPlane(centerScreenX - halfSpan2Px * vX, centerScreenY - halfSpan2Px * vY, sWidth, sHeight, invVpMatrix, midPlaneY)

        if (pCenterMid == null || p1AMid == null || p1BMid == null || p2AMid == null || p2BMid == null) return null

        val v1X = p1AMid[0] - p1BMid[0]; val v1Z = p1AMid[2] - p1BMid[2]
        val span1 = sqrt(v1X * v1X + v1Z * v1Z).coerceIn(0.035f, 0.45f)
        val u1X = v1X / span1; val u1Z = v1Z / span1

        val v2X = p2AMid[0] - p2BMid[0]; val v2Z = p2AMid[2] - p2BMid[2]
        val span2 = sqrt(v2X * v2X + v2Z * v2Z).coerceIn(0.035f, 0.45f)
        val u2X = v2X / span2; val u2Z = v2Z / span2

        val camAngle = atan2(camX - pCenterMid[0], camZ - pCenterMid[2])
        val camDirX = sin(camAngle); val camDirZ = cos(camAngle)

        val dot0 = u1X * camDirX + u1Z * camDirZ
        val dot1 = -dot0
        val dot2 = u2X * camDirX + u2Z * camDirZ
        val dot3 = -dot2

        val dots = floatArrayOf(dot0, dot1, dot2, dot3)
        var bestIdx = 0
        var maxDot = -Float.MAX_VALUE
        for (i in 0 until 4) {
            if (dots[i] > maxDot) { maxDot = dots[i]; bestIdx = i }
        }

        val margin = 1.06f
        val finalWidth: Float
        val finalDepth: Float
        val headingAngle: Float

        when (bestIdx) {
            0 -> { finalDepth = (span1 * margin + 0.006f).coerceIn(0.04f, 0.45f); finalWidth = (span2 * margin + 0.006f).coerceIn(0.04f, 0.45f); headingAngle = atan2(u1X, u1Z) }
            1 -> { finalDepth = (span1 * margin + 0.006f).coerceIn(0.04f, 0.45f); finalWidth = (span2 * margin + 0.006f).coerceIn(0.04f, 0.45f); headingAngle = atan2(-u1X, -u1Z) }
            2 -> { finalDepth = (span2 * margin + 0.006f).coerceIn(0.04f, 0.45f); finalWidth = (span1 * margin + 0.006f).coerceIn(0.04f, 0.45f); headingAngle = atan2(u2X, u2Z) }
            else -> { finalDepth = (span2 * margin + 0.006f).coerceIn(0.04f, 0.45f); finalWidth = (span1 * margin + 0.006f).coerceIn(0.04f, 0.45f); headingAngle = atan2(-u2X, -u2Z) }
        }

        return MagicTouch3DFit(
            worldX = pCenterMid[0],
            worldZ = pCenterMid[2],
            widthM = finalWidth,
            depthM = finalDepth,
            headingRad = headingAngle,
            screenCX = centerScreenX,
            screenCY = centerScreenY
        )
    }

    private fun unprojectScreenRay(
        screenX: Float, screenY: Float,
        screenWidth: Float, screenHeight: Float,
        invVpMatrix: FloatArray
    ): FloatArray? {
        val ndcX = (screenX / screenWidth) * 2f - 1f
        val ndcY = 1f - (screenY / screenHeight) * 2f

        val nearVec = floatArrayOf(ndcX, ndcY, -1f, 1f)
        val farVec = floatArrayOf(ndcX, ndcY, 1f, 1f)
        val nearWorld = FloatArray(4)
        val farWorld = FloatArray(4)

        Matrix.multiplyMV(nearWorld, 0, invVpMatrix, 0, nearVec, 0)
        Matrix.multiplyMV(farWorld, 0, invVpMatrix, 0, farVec, 0)

        val wNear = nearWorld[3]
        val wFar = farWorld[3]
        if (abs(wNear) < 1e-5f || abs(wFar) < 1e-5f) return null

        val rOx = nearWorld[0] / wNear
        val rOy = nearWorld[1] / wNear
        val rOz = nearWorld[2] / wNear

        val dirX = (farWorld[0] / wFar) - rOx
        val dirY = (farWorld[1] / wFar) - rOy
        val dirZ = (farWorld[2] / wFar) - rOz
        val len = sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ)
        if (len < 1e-4f) return null

        return floatArrayOf(rOx, rOy, rOz, dirX / len, dirY / len, dirZ / len)
    }

    private fun intersectScreenRayWithHorizontalPlane(
        screenX: Float, screenY: Float,
        screenWidth: Float, screenHeight: Float,
        invVpMatrix: FloatArray,
        planeY: Float
    ): FloatArray? {
        val ray = unprojectScreenRay(screenX, screenY, screenWidth, screenHeight, invVpMatrix) ?: return null
        if (abs(ray[4]) < 1e-4f) return null
        val t = (planeY - ray[1]) / ray[4]
        if (t <= 0.05f) return null
        return floatArrayOf(ray[0] + t * ray[3], planeY, ray[2] + t * ray[5])
    }

    // =========================================================================
    // 60 FPS PER-FRAME ARCORE TRACKING LOOP
    // =========================================================================
    private fun onTrackingFrame(frame: Frame) {
        val camera = frame.camera
        if (camera.trackingState != TrackingState.TRACKING) return

        camera.getViewMatrix(frameViewMatrix, 0)
        camera.getProjectionMatrix(frameProjMatrix, 0, 0.05f, 50.0f)
        Matrix.multiplyMM(frameVpMatrix, 0, frameProjMatrix, 0, frameViewMatrix, 0)

        val camPoseNow = camera.pose
        liveCamX = camPoseNow.tx()
        liveCamY = camPoseNow.ty()
        liveCamZ = camPoseNow.tz()
        liveCamFwdX = -camPoseNow.zAxis[0]
        liveCamFwdY = -camPoseNow.zAxis[1]
        liveCamFwdZ = -camPoseNow.zAxis[2]

        val now = System.currentTimeMillis()
        val shouldUpdateHud = (now - lastHudUpdateMs) >= 100L

        if (!isBoxPlaced) {
            val (tablePlane, tablePose) = resolveStrictTableSurface(frame)

            if (tablePlane != null && tablePose != null) {
                val tableY = tablePlane.centerPose.ty()
                val aimX = tablePose.tx()
                val aimZ = tablePose.tz()

                if (hasLiveCling) {
                    // World-pinned 60 FPS tracking via ARCore preLockAnchor (zero ML calls while orbiting!)
                    val pin = preLockAnchor
                    if (pin != null) {
                        if (pin.trackingState == TrackingState.TRACKING) {
                            val stablePin = getStabilizedAnchorPose(pin.pose)
                            liveWorldX = stablePin.tx()
                            liveWorldY = tableY
                            liveWorldZ = stablePin.tz()
                        } else if (pin.trackingState == TrackingState.STOPPED) {
                            releasePreLockPin()
                        }
                    } else {
                        liveWorldY = tableY
                    }

                    // Instant release back to blue 60 FPS scanner if user pans crosshair away from pinned object
                    if (hasLiveCling) {
                        val panDist = sqrt((aimX - liveWorldX).pow(2) + (aimZ - liveWorldZ).pow(2))
                        val releaseRadius = max(max(liveWidth, liveDepth) * 0.85f, 0.11f)
                        if (panDist > releaseRadius) {
                            releasePreLockPin()
                        }
                    }
                } else if (!isAutoSnapRunning) {
                    // 60 FPS Dwell Detector: fires MagicTouch once when aim settles on an object
                    val skipFailedSpot = hasFailedSpot &&
                        sqrt((aimX - failedSpotX).pow(2) + (aimZ - failedSpotZ).pow(2)) < 0.045f &&
                        (now - failedSpotTimeMs) < 1500L

                    if (skipFailedSpot) {
                        hasDwellRef = false
                        dwellProgress = 0f
                    } else if (!hasDwellRef) {
                        hasDwellRef = true
                        dwellStartMs = now
                        dwellRefX = aimX
                        dwellRefZ = aimZ
                        dwellFwdX = liveCamFwdX
                        dwellFwdY = liveCamFwdY
                        dwellFwdZ = liveCamFwdZ
                        dwellProgress = 0f
                    } else {
                        val moveDist = sqrt((aimX - dwellRefX).pow(2) + (aimZ - dwellRefZ).pow(2))
                        val rotDot = liveCamFwdX * dwellFwdX + liveCamFwdY * dwellFwdY + liveCamFwdZ * dwellFwdZ

                        if (moveDist > 0.022f || rotDot < 0.9985f) {
                            // Still sweeping -> keep resetting dwell smoothly at 60 FPS
                            dwellStartMs = now
                            dwellRefX = aimX
                            dwellRefZ = aimZ
                            dwellFwdX = liveCamFwdX
                            dwellFwdY = liveCamFwdY
                            dwellFwdZ = liveCamFwdZ
                            dwellProgress = 0f
                        } else {
                            val elapsed = now - dwellStartMs
                            dwellProgress = (elapsed.toFloat() / dwellDurationMs).coerceIn(0f, 1f)
                            if (elapsed >= dwellDurationMs && magicTouchInterpreter != null) {
                                triggerAutoSnapMagicTouch(frame, tablePose)
                            }
                        }
                    }
                }
            } else {
                releasePreLockPin()
            }

            overlayView.updatePreLockSnapshot(frameVpMatrix, tablePose, dwellProgress, isAutoSnapRunning)
            requestOverlayRedraw()

            if (shouldUpdateHud) {
                lastHudUpdateMs = now
                if (lastSnapLatencyMs > 0L) {
                    setTextIfChanged(valAlign, "SNAP: ${lastSnapLatencyMs}ms")
                }
                if (hasLiveCling && tablePlane != null) {
                    val wCm = (liveWidth * 100).roundToInt()
                    val dCm = (liveDepth * 100).roundToInt()
                    val hCm = (liveHeight * 100).roundToInt()
                    setTextIfChanged(distIndicator, "◎ PINNED FIT: ${wCm}×${dCm}×${hCm}cm — TAP LOCK", Color.parseColor("#3FB950"))
                    setTextIfChanged(valScale, "RING: ${wCm}×${dCm}")
                } else if (isAutoSnapRunning && tablePlane != null) {
                    setTextIfChanged(distIndicator, "◎ SNAPPING OBJECT...", Color.parseColor("#E3B341"))
                } else if (tablePlane != null) {
                    setTextIfChanged(distIndicator, "TABLE LOCKED: AIM & PAUSE ON ASSET", Color.parseColor("#58A6FF"))
                } else {
                    setTextIfChanged(distIndicator, "SCANNING FOR TABLE PLANE...", Color.parseColor("#E3B341"))
                }
            }
            return
        }

        val anchor = prismAnchor ?: return
        if (anchor.trackingState != TrackingState.TRACKING) return

        val stablePose = getStabilizedAnchorPose(anchor.pose)
        val camPose = camera.pose

        overlayView.updateLockedSnapshot(frameVpMatrix, stablePose, camPose)

        if (currentStepIdx >= sequence.size) {
            requestOverlayRedraw()
            return
        }

        val step = sequence[currentStepIdx]

        val desc = getFacetDescriptor(currentStepIdx)
        val faceCenterWorld = localToWorld(desc.centerX, desc.centerY, desc.centerZ, stablePose)
        val faceNormalWorld = normalToWorld(desc.normX, desc.normY, desc.normZ, stablePose)

        val camToFaceX = camPose.tx() - faceCenterWorld[0]
        val camToFaceY = camPose.ty() - faceCenterWorld[1]
        val camToFaceZ = camPose.tz() - faceCenterWorld[2]
        val facingDot = camToFaceX * faceNormalWorld[0] + camToFaceY * faceNormalWorld[1] + camToFaceZ * faceNormalWorld[2]
        val isFacingCamera = facingDot > 0.001f

        val toFaceVecX = faceCenterWorld[0] - camPose.tx()
        val toFaceVecY = faceCenterWorld[1] - camPose.ty()
        val toFaceVecZ = faceCenterWorld[2] - camPose.tz()
        val distMeters = sqrt(toFaceVecX * toFaceVecX + toFaceVecY * toFaceVecY + toFaceVecZ * toFaceVecZ)
        val distCm = distMeters * 100f
        val isDistanceOptimal = distCm in 20.0f..38.0f

        val camForward = floatArrayOf(-camPose.zAxis[0], -camPose.zAxis[1], -camPose.zAxis[2])
        val actualTiltDeg = Math.toDegrees(asin((-camForward[1]).coerceIn(-1.0f, 1.0f).toDouble())).roundToInt()

        val dotNormal = -(camForward[0] * faceNormalWorld[0] + camForward[1] * faceNormalWorld[1] + camForward[2] * faceNormalWorld[2])
        val normAngleErr = Math.toDegrees(acos(dotNormal.coerceIn(-1.0f, 1.0f).toDouble())).toFloat()
        val isNormalAligned = isFacingCamera && (normAngleErr <= 10.0f)

        val safeDist = if (distMeters > 0.001f) distMeters else 1.0f
        val toFaceDirX = toFaceVecX / safeDist
        val toFaceDirY = toFaceVecY / safeDist
        val toFaceDirZ = toFaceVecZ / safeDist
        val dotCenter = camForward[0] * toFaceDirX + camForward[1] * toFaceDirY + camForward[2] * toFaceDirZ
        val centAngleErr = Math.toDegrees(acos(dotCenter.coerceIn(-1.0f, 1.0f).toDouble())).toFloat()
        val isCentered = isFacingCamera && (centAngleErr <= 12.0f)

        val isReadyToCapture = isNormalAligned && isCentered && isDistanceOptimal

        if (isReadyToCapture && !isCapturing && now >= cooldownUntil) {
            overlayView.isAligned = true
            if (alignStartTime == null) {
                alignStartTime = now
            } else if (now - alignStartTime!! >= 450) {
                alignStartTime = null
                triggerLocalCapture()
            }
        } else {
            if (!isCapturing) {
                overlayView.isAligned = false
                alignStartTime = null
            }
        }

        requestOverlayRedraw()

        if (shouldUpdateHud) {
            lastHudUpdateMs = now
            setTextIfChanged(valDist, "DIST: ${distCm.roundToInt()}cm")
            setTextIfChanged(valAlign, "TILT: ${actualTiltDeg}° [${step.targetTilt}°]")

            if (distCm < 20f) {
                setTextIfChanged(distIndicator, "⚠️ TOO CLOSE (${distCm.roundToInt()}cm) — STEP BACK", Color.parseColor("#F85149"))
            } else if (distCm > 38f) {
                setTextIfChanged(distIndicator, "MOVE CLOSER (${distCm.roundToInt()}cm) — AIM 25-32cm", Color.parseColor("#E3B341"))
            } else {
                setTextIfChanged(distIndicator, "✓ FOCUS DISTANCE: ${distCm.roundToInt()}cm (SHARP)", Color.parseColor("#3FB950"))
            }

            if (!isFacingCamera) {
                setTextIfChanged(normalIndicator, "MOVE TO FRONT OF ${step.label}", Color.parseColor("#F85149"))
                setTextIfChanged(centerIndicator, "FACET FACING AWAY", Color.parseColor("#8B949E"))
            } else {
                if (isNormalAligned) {
                    setTextIfChanged(normalIndicator, "TILT: ${actualTiltDeg}° | ALIGN: ${String.format(Locale.US, "%.1f", normAngleErr)}° (LOCKED)", Color.parseColor("#3FB950"))
                } else {
                    setTextIfChanged(normalIndicator, "TILT: ${actualTiltDeg}° [AIM ${step.targetTilt}°] | DEV: ${String.format(Locale.US, "%.1f", normAngleErr)}°", Color.parseColor("#F85149"))
                }

                if (isCentered) {
                    setTextIfChanged(centerIndicator, "BULLSEYE CENTERED (OFFSET: ${String.format(Locale.US, "%.1f", centAngleErr)}°)", Color.parseColor("#3FB950"))
                } else {
                    setTextIfChanged(centerIndicator, "AIM AT RETICLE (OFFSET: ${String.format(Locale.US, "%.1f", centAngleErr)}°)", Color.parseColor("#8B949E"))
                }
            }
        }
    }

    private fun triggerLocalCapture() {
        if (isCapturing || currentStepIdx >= sequence.size) return
        isCapturing = true

        val snappedStepIdx = currentStepIdx
        val step = sequence[snappedStepIdx]

        playSnapFeedback()

        val bitmap = Bitmap.createBitmap(sceneView.width, sceneView.height, Bitmap.Config.ARGB_8888)
        PixelCopy.request(
            sceneView as SurfaceView,
            bitmap,
            PixelCopy.OnPixelCopyFinishedListener { copyResult ->
                if (copyResult == PixelCopy.SUCCESS) {
                    Thread {
                        try {
                            val file = File(cacheDir, "plate_${step.id}.jpg")
                            val fos = FileOutputStream(file)
                            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, fos)
                            fos.flush()
                            fos.close()

                            val thumb = Bitmap.createScaledBitmap(bitmap, 128, 128, false)
                            capturedThumbnails[snappedStepIdx] = thumb

                            runOnUiThread {
                                previewCardViews[snappedStepIdx].setImageBitmap(thumb)

                                currentStepIdx++
                                isCapturing = false
                                cooldownUntil = System.currentTimeMillis() + 800L
                                alignStartTime = null
                                updateChecklistUI()

                                if (currentStepIdx >= sequence.size) {
                                    targetBadge.text = "✓ ALL 10 PLATES STORED LOCALLY"
                                    targetBadge.setTextColor(Color.parseColor("#3FB950"))
                                    distIndicator.text = "CAPTURE COMPLETE"
                                    distIndicator.setTextColor(Color.parseColor("#3FB950"))
                                    normalIndicator.text = "READY FOR MODAL BATCH UPLOAD"
                                    centerIndicator.text = "TAP UPLOAD BELOW"
                                    forceSnapBtn.visibility = View.GONE
                                    scaleBar.visibility = View.GONE
                                } else {
                                    targetBadge.text = sequence[currentStepIdx].label
                                    forceSnapBtn.text = "FORCE SNAP [${sequence[currentStepIdx].id.uppercase(Locale.US)}]"
                                }
                                requestOverlayRedraw()
                            }
                        } catch (e: Exception) {
                            Log.e("Capture", "Error saving plate locally", e)
                            isCapturing = false
                        }
                    }.start()
                } else {
                    isCapturing = false
                }
            },
            mainHandler
        )
    }

    private fun startBatchUpload() {
        isBatchUploading = true
        batchUploadBtn.isEnabled = false
        batchUploadBtn.text = "UPLOADING 0/${capturedThumbnails.size}..."

        Thread {
            val total = capturedThumbnails.size
            var uploaded = 0

            for (entry in capturedThumbnails.entries.sortedBy { it.key }) {
                val idx = entry.key
                val step = sequence[idx]
                val file = File(cacheDir, "plate_${step.id}.jpg")
                if (!file.exists()) continue

                val bytes = file.readBytes()
                val b64 = Base64.encodeToString(bytes, Base64.NO_WRAP)

                val json = JSONObject().apply {
                    put("target_id", step.id)
                    put("b64", b64)
                    put("dimensions", JSONObject().apply {
                        put("w", prismWidth)
                        put("h", prismHeight)
                        put("d", prismDepth)
                    })
                }

                val body = json.toString().toRequestBody("application/json".toMediaType())
                val request = Request.Builder().url(modalEndpoint).post(body).build()

                try {
                    val response = httpClient.newCall(request).execute()
                    response.close()
                    uploaded++
                    runOnUiThread {
                        batchUploadBtn.text = "UPLOADING $uploaded/$total..."
                    }
                } catch (e: Exception) {
                    Log.e("BatchUpload", "Failed on ${step.id}", e)
                }
            }

            runOnUiThread {
                isBatchUploading = false
                batchUploadBtn.isEnabled = true
                batchUploadBtn.text = "✓ ALL $uploaded PLATES ON MODAL VOLUME"
                batchUploadBtn.setBackgroundColor(Color.parseColor("#238636"))
                Toast.makeText(this, "Batch upload complete ($uploaded plates)", Toast.LENGTH_LONG).show()
            }
        }.start()
    }

    private fun playSnapFeedback() {
        try {
            toneGen?.startTone(ToneGenerator.TONE_PROP_BEEP, 120)
        } catch (e: Exception) {}

        snapFlash.alpha = 0.5f
        snapFlash.visibility = View.VISIBLE
        snapFlash.animate()
            .alpha(0f)
            .setDuration(180)
            .withEndAction { snapFlash.visibility = View.GONE }
            .start()
    }

    data class FacetDesc(
        val centerX: Float, val centerY: Float, val centerZ: Float,
        val normX: Float, val normY: Float, val normZ: Float,
        val isChamfer: Boolean
    )

    private fun getFacetDescriptor(step: Int): FacetDesc {
        val cos30 = cos(Math.toRadians(30.0)).toFloat()
        val sin30 = sin(Math.toRadians(30.0)).toFloat()
        val sqrt2Inv = 0.7071f
        val hW = prismWidth / 2f
        val hD = prismDepth / 2f
        val hMid = prismHeight * 0.50f
        val hChamferY = hMid + (prismHeight - hMid) * 0.50f

        return when (step) {
            0 -> FacetDesc(0f, hMid * 0.5f, hD, 0f, 0f, 1f, false)
            1 -> FacetDesc(hW * 0.76f, hChamferY, hD * 0.76f, cos30 * sqrt2Inv, sin30, cos30 * sqrt2Inv, true)
            2 -> FacetDesc(hW, hMid * 0.5f, 0f, 1f, 0f, 0f, false)
            3 -> FacetDesc(hW * 0.76f, hChamferY, -hD * 0.76f, cos30 * sqrt2Inv, sin30, -cos30 * sqrt2Inv, true)
            4 -> FacetDesc(0f, hMid * 0.5f, -hD, 0f, 0f, -1f, false)
            5 -> FacetDesc(-hW * 0.76f, hChamferY, -hD * 0.76f, -cos30 * sqrt2Inv, sin30, -cos30 * sqrt2Inv, true)
            6 -> FacetDesc(-hW, hMid * 0.5f, 0f, -1f, 0f, 0f, false)
            7 -> FacetDesc(-hW * 0.76f, hChamferY, hD * 0.76f, -cos30 * sqrt2Inv, sin30, -cos30 * sqrt2Inv, true)
            8 -> FacetDesc(0f, prismHeight, 0f, 0f, 1f, 0f, false)
            else -> FacetDesc(0f, 0f, 0f, 0f, -1f, 0f, false)
        }
    }

    private fun rotateByAzimuth(x: Float, y: Float, z: Float, azimuth: Float = initialAzimuth): FloatArray {
        val c = cos(azimuth)
        val s = sin(azimuth)
        return floatArrayOf(x * c + z * s, y, -x * s + z * c)
    }

    private fun localToWorld(lx: Float, ly: Float, lz: Float, anchorPose: Pose): FloatArray {
        val rotated = rotateByAzimuth(lx, ly, lz, initialAzimuth)
        return floatArrayOf(
            anchorPose.tx() + rotated[0],
            anchorPose.ty() + rotated[1],
            anchorPose.tz() + rotated[2]
        )
    }

    private fun liveLocalToWorld(lx: Float, ly: Float, lz: Float): FloatArray {
        val rotated = rotateByAzimuth(lx, ly, lz, liveAzimuth)
        return floatArrayOf(liveWorldX + rotated[0], liveWorldY + rotated[1], liveWorldZ + rotated[2])
    }

    private fun normalToWorld(nx: Float, ny: Float, nz: Float, anchorPose: Pose): FloatArray {
        return rotateByAzimuth(nx, ny, nz, initialAzimuth)
    }

    inner class PrismOverlayView(context: Context) : View(context) {

        var isAligned = false

        private var hasSnapshot = false
        private val vpMatrix = FloatArray(16)
        private var snapAnchorPose: Pose? = null
        private var snapCamPose: Pose? = null
        private var snapHitPose: Pose? = null
        private var snapDwellProgress = 0f
        private var snapIsSnapping = false

        private val scratchPath = Path()
        private val scratchClipVec = FloatArray(4)
        private val scratchWorldVec = FloatArray(4)
        private val dwellArcRect = RectF()

        fun updatePreLockSnapshot(vpMat: FloatArray, hitPose: Pose?, dwellProg: Float, snapping: Boolean) {
            System.arraycopy(vpMat, 0, vpMatrix, 0, 16)
            snapHitPose = hitPose
            snapDwellProgress = dwellProg
            snapIsSnapping = snapping
            hasSnapshot = true
        }

        fun updateLockedSnapshot(vpMat: FloatArray, anchorPose: Pose, camPose: Pose) {
            System.arraycopy(vpMat, 0, vpMatrix, 0, 16)
            snapAnchorPose = anchorPose
            snapCamPose = camPose
            hasSnapshot = true
        }

        private val wirePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 2.5f
        }

        private val perforatedMeshPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
        }

        private val reticlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 5f
        }

        private val fillReticlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
        }

        private val centerGuidePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(140, 255, 255, 255)
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }

        private val centerDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(220, 255, 255, 255)
            style = Paint.Style.FILL
        }

        private val dwellArcPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#3FB950")
            style = Paint.Style.STROKE
            strokeWidth = 5f
            strokeCap = Paint.Cap.ROUND
        }

        private val webXrDiskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(40, 88, 166, 255)
            style = Paint.Style.FILL
        }

        private val webXrWallBandPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(42, 63, 185, 80)
            style = Paint.Style.FILL
        }

        private val webXrRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#3FB950")
            style = Paint.Style.STROKE
            strokeWidth = 5.0f
        }

        private val webXrTopRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(230, 88, 235, 110)
            style = Paint.Style.STROKE
            strokeWidth = 3.8f
        }

        private val webXrInnerRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(180, 255, 255, 255)
            style = Paint.Style.STROKE
            strokeWidth = 2.2f
        }

        private val webXrRibPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(150, 63, 185, 80)
            style = Paint.Style.STROKE
            strokeWidth = 2.4f
        }

        private val redMeshShader = createPerforatedShader(Color.parseColor("#F85149"))
        private val greenMeshShader = createPerforatedShader(Color.parseColor("#3FB950"))
        private val greyMeshShader = createPerforatedShader(Color.parseColor("#8B949E"))

        private fun createPerforatedShader(baseColor: Int): BitmapShader {
            val tileSize = 24
            val bmp = Bitmap.createBitmap(tileSize, tileSize, Bitmap.Config.ARGB_8888)
            val cv = Canvas(bmp)

            val bodyPaint = Paint().apply {
                color = Color.argb(175, Color.red(baseColor), Color.green(baseColor), Color.blue(baseColor))
                style = Paint.Style.FILL
            }
            cv.drawRect(0f, 0f, tileSize.toFloat(), tileSize.toFloat(), bodyPaint)

            val clearHolePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
            }
            val holeR = 5.2f
            cv.drawCircle(tileSize / 2f, tileSize / 2f, holeR, clearHolePaint)
            cv.drawCircle(0f, 0f, holeR, clearHolePaint)
            cv.drawCircle(tileSize.toFloat(), 0f, holeR, clearHolePaint)
            cv.drawCircle(0f, tileSize.toFloat(), holeR, clearHolePaint)
            cv.drawCircle(tileSize.toFloat(), tileSize.toFloat(), holeR, clearHolePaint)

            val rimPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.argb(90, 255, 255, 255)
                style = Paint.Style.STROKE
                strokeWidth = 1.0f
            }
            cv.drawCircle(tileSize / 2f, tileSize / 2f, holeR, rimPaint)
            cv.drawCircle(0f, 0f, holeR, rimPaint)
            cv.drawCircle(tileSize.toFloat(), 0f, holeR, rimPaint)
            cv.drawCircle(0f, tileSize.toFloat(), holeR, rimPaint)
            cv.drawCircle(tileSize.toFloat(), tileSize.toFloat(), holeR, rimPaint)

            return BitmapShader(bmp, Shader.TileMode.REPEAT, Shader.TileMode.REPEAT)
        }

        private fun getMeshShaderForColor(color: Int): BitmapShader {
            return when (color) {
                Color.parseColor("#3FB950") -> greenMeshShader
                Color.parseColor("#8B949E") -> greyMeshShader
                else -> redMeshShader
            }
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            if (!hasSnapshot) return

            val screenW = width.toFloat()
            val screenH = height.toFloat()

            // =========================================================================
            // PRE-LOCK PHASE: 60 FPS TABLE-GROUNDED RETICLE + WORLD-PINNED AUTO-SNAP
            // =========================================================================
            if (!isBoxPlaced) {
                val cx = screenW / 2f
                val cy = screenH / 2f
                canvas.drawCircle(cx, cy, 18f, centerGuidePaint)
                canvas.drawCircle(cx, cy, 4.5f, centerDotPaint)

                // Draw dwell / snapping progress arc around center aim dot
                if (!hasLiveCling && (snapDwellProgress > 0.05f || snapIsSnapping)) {
                    dwellArcRect.set(cx - 22f, cy - 22f, cx + 22f, cy + 22f)
                    dwellArcPaint.color = if (snapIsSnapping) Color.parseColor("#E3B341") else Color.parseColor("#3FB950")
                    val sweep = if (snapIsSnapping) 360f else (snapDwellProgress * 360f)
                    canvas.drawArc(dwellArcRect, -90f, sweep, false, dwellArcPaint)
                }

                if (hasLiveCling && snapHitPose != null) {
                    // Squarer 0.58-exponent squircle at 0.62x clears all rectangular corners cleanly
                    val rx = (liveWidth * 0.62f).coerceIn(0.028f, 0.28f)
                    val rz = (liveDepth * 0.62f).coerceIn(0.028f, 0.28f)
                    val ringH = liveHeight.coerceIn(0.028f, 0.24f)

                    val basePts = mutableListOf<PointF?>()
                    val midPts = mutableListOf<PointF?>()
                    val topPts = mutableListOf<PointF?>()
                    val innerBasePts = mutableListOf<PointF?>()
                    val steps = 36

                    for (i in 0 until steps) {
                        val ang = (2.0 * Math.PI * i / steps)
                        val c = cos(ang)
                        val s = sin(ang)
                        val lx = (rx * c.sign * abs(c).pow(0.58)).toFloat()
                        val lz = (rz * s.sign * abs(s).pow(0.58)).toFloat()

                        basePts.add(projectPoint(liveLocalToWorld(lx, 0f, lz), screenW, screenH))
                        midPts.add(projectPoint(liveLocalToWorld(lx, ringH * 0.5f, lz), screenW, screenH))
                        topPts.add(projectPoint(liveLocalToWorld(lx, ringH, lz), screenW, screenH))
                        innerBasePts.add(projectPoint(liveLocalToWorld(lx * 0.86f, 0f, lz * 0.86f), screenW, screenH))
                    }

                    if (basePts.all { it != null } && topPts.all { it != null } && midPts.all { it != null }) {
                        for (i in 0 until steps) {
                            val next = (i + 1) % steps
                            scratchPath.reset()
                            scratchPath.moveTo(basePts[i]!!.x, basePts[i]!!.y)
                            scratchPath.lineTo(basePts[next]!!.x, basePts[next]!!.y)
                            scratchPath.lineTo(topPts[next]!!.x, topPts[next]!!.y)
                            scratchPath.lineTo(topPts[i]!!.x, topPts[i]!!.y)
                            scratchPath.close()
                            canvas.drawPath(scratchPath, webXrWallBandPaint)
                        }

                        for (i in 0 until steps step 2) {
                            canvas.drawLine(
                                basePts[i]!!.x, basePts[i]!!.y,
                                topPts[i]!!.x, topPts[i]!!.y,
                                if (i % 9 == 0) webXrRingPaint else webXrRibPaint
                            )
                        }

                        for (i in 0 until steps step 2) {
                            val next = (i + 1) % steps
                            canvas.drawLine(midPts[i]!!.x, midPts[i]!!.y, midPts[next]!!.x, midPts[next]!!.y, webXrInnerRingPaint)
                        }

                        webXrRingPaint.color = Color.parseColor("#3FB950")
                        for (i in 0 until steps) {
                            val next = (i + 1) % steps
                            canvas.drawLine(basePts[i]!!.x, basePts[i]!!.y, basePts[next]!!.x, basePts[next]!!.y, webXrRingPaint)
                            if (innerBasePts[i] != null && innerBasePts[next] != null && i % 2 == 0) {
                                canvas.drawLine(innerBasePts[i]!!.x, innerBasePts[i]!!.y, innerBasePts[next]!!.x, innerBasePts[next]!!.y, webXrInnerRingPaint)
                            }
                        }

                        for (i in 0 until steps) {
                            val next = (i + 1) % steps
                            canvas.drawLine(topPts[i]!!.x, topPts[i]!!.y, topPts[next]!!.x, topPts[next]!!.y, webXrTopRingPaint)
                        }
                    }
                    return
                }

                val hp = snapHitPose
                if (hp != null) {
                    val diskRadius = 0.075f
                    val ringH = 0.035f
                    val bPts = mutableListOf<PointF?>()
                    val tPts = mutableListOf<PointF?>()

                    webXrDiskPaint.color = Color.argb(35, 88, 166, 255)
                    webXrRingPaint.color = if (snapIsSnapping) Color.parseColor("#E3B341") else Color.parseColor("#58A6FF")

                    for (i in 0 until 32) {
                        val ang = (2.0 * Math.PI * i / 32).toFloat()
                        val wx = hp.tx() + diskRadius * cos(ang)
                        val wz = hp.tz() + diskRadius * sin(ang)
                        bPts.add(projectPoint(wx, hp.ty(), wz, screenW, screenH))
                        tPts.add(projectPoint(wx, hp.ty() + ringH, wz, screenW, screenH))
                    }

                    if (bPts.all { it != null } && tPts.all { it != null }) {
                        for (i in bPts.indices) {
                            val next = (i + 1) % bPts.size
                            canvas.drawLine(bPts[i]!!.x, bPts[i]!!.y, bPts[next]!!.x, bPts[next]!!.y, webXrRingPaint)
                            canvas.drawLine(tPts[i]!!.x, tPts[i]!!.y, tPts[next]!!.x, tPts[next]!!.y, webXrInnerRingPaint)
                            if (i % 4 == 0) {
                                canvas.drawLine(bPts[i]!!.x, bPts[i]!!.y, tPts[i]!!.x, tPts[i]!!.y, webXrInnerRingPaint)
                            }
                        }
                    }
                }
                return
            }

            val anchorPose = snapAnchorPose ?: return
            val camPose = snapCamPose ?: return

            canvas.drawCircle(screenW / 2f, screenH / 2f, 75f, centerGuidePaint)
            canvas.drawCircle(screenW / 2f, screenH / 2f, 8f, centerGuidePaint)

            val hW = prismWidth / 2f
            val hD = prismDepth / 2f
            val r = min(hW, hD) * 0.36f
            val hMid = prismHeight * 0.50f
            val tan30 = tan(Math.toRadians(30.0)).toFloat()
            val dChamfer = min((prismHeight - hMid) * tan30, r * 0.82f)
            val rTop = max(r - dChamfer * 0.45f, 0.02f)

            val baseLoop = generateFilletedSquircleLoop(hW, hD, r)
            val midLoop = generateFilletedSquircleLoop(hW, hD, r)
            val topLoop = generateFilletedSquircleLoop(hW - dChamfer, hD - dChamfer, rTop)

            val greyColor = Color.parseColor("#8B949E")
            val activeRed = Color.parseColor("#F85149")
            val capturedGreen = Color.parseColor("#3FB950")

            for (sector in 0 until 4) {
                val facetIdx = sector * 2
                val isCaptured = capturedThumbnails.containsKey(facetIdx)
                val isActiveStep = currentStepIdx == facetIdx
                val wallColor = when {
                    isCaptured -> capturedGreen
                    isActiveStep && isAligned -> capturedGreen
                    else -> activeRed
                }
                drawFilletedWallSector(canvas, sector, baseLoop, midLoop, 0f, hMid, anchorPose, screenW, screenH, wallColor, true)
                drawFilletedCornerSector(canvas, sector, baseLoop, midLoop, 0f, hMid, anchorPose, screenW, screenH, greyColor, false)
            }

            for (sector in 0 until 4) {
                drawFilletedWallSector(canvas, sector, midLoop, topLoop, hMid, prismHeight, anchorPose, screenW, screenH, greyColor, false)

                val facetIdx = sector * 2 + 1
                val isCaptured = capturedThumbnails.containsKey(facetIdx)
                val isActiveStep = currentStepIdx == facetIdx
                val chamferColor = when {
                    isCaptured -> capturedGreen
                    isActiveStep && isAligned -> capturedGreen
                    else -> activeRed
                }
                drawFilletedCornerSector(canvas, sector, midLoop, topLoop, hMid, prismHeight, anchorPose, screenW, screenH, chamferColor, true)
            }

            val isTopCaptured = capturedThumbnails.containsKey(8)
            val isTopActiveStep = currentStepIdx == 8
            val topColor = when {
                isTopCaptured -> capturedGreen
                isTopActiveStep && isAligned -> capturedGreen
                else -> activeRed
            }
            drawTopCap(canvas, topLoop, prismHeight, anchorPose, screenW, screenH, topColor, true)

            if (currentStepIdx < sequence.size) {
                val desc = getFacetDescriptor(currentStepIdx)
                val faceCenterWorld = localToWorld(desc.centerX, desc.centerY, desc.centerZ, anchorPose)
                val faceNormalWorld = normalToWorld(desc.normX, desc.normY, desc.normZ, anchorPose)

                val camToFaceX = camPose.tx() - faceCenterWorld[0]
                val camToFaceY = camPose.ty() - faceCenterWorld[1]
                val camToFaceZ = camPose.tz() - faceCenterWorld[2]
                val facingDot = camToFaceX * faceNormalWorld[0] + camToFaceY * faceNormalWorld[1] + camToFaceZ * faceNormalWorld[2]

                if (facingDot > 0.001f) {
                    drawPlanar3DReticle(canvas, desc, anchorPose, screenW, screenH)
                }
            }
        }

        private fun generateFilletedSquircleLoop(hW: Float, hD: Float, r: Float): List<FloatArray> {
            val loop = mutableListOf<FloatArray>()
            val xS = hW - r
            val zS = hD - r

            for (quad in 0 until 4) {
                when (quad) {
                    0 -> { loop.add(floatArrayOf(-xS, 0f, hD)); loop.add(floatArrayOf(xS, 0f, hD)) }
                    1 -> { loop.add(floatArrayOf(hW, 0f, zS)); loop.add(floatArrayOf(hW, 0f, -zS)) }
                    2 -> { loop.add(floatArrayOf(xS, 0f, -hD)); loop.add(floatArrayOf(-xS, 0f, -hD)) }
                    3 -> { loop.add(floatArrayOf(-hW, 0f, -zS)); loop.add(floatArrayOf(-hW, 0f, zS)) }
                }

                val cX = if (quad == 0 || quad == 1) xS else -xS
                val cZ = if (quad == 0 || quad == 3) zS else -zS
                val startAngle = Math.toRadians((90.0 - quad * 90.0))

                for (step in 1..4) {
                    val angle = startAngle - (Math.PI / 2.0) * (step / 5.0)
                    val px = cX + r * cos(angle).toFloat()
                    val pz = cZ + r * sin(angle).toFloat()
                    loop.add(floatArrayOf(px, 0f, pz))
                }
            }
            return loop
        }

        private fun drawFilletedWallSector(
            canvas: Canvas, sector: Int,
            base: List<FloatArray>, top: List<FloatArray>,
            y0: Float, y1: Float,
            anchorPose: Pose,
            w: Float, h: Float, color: Int, isActive: Boolean
        ) {
            val i0 = sector * 6
            val i1 = (i0 + 1) % base.size

            val p0 = projectPoint(localToWorld(base[i0][0], y0, base[i0][2], anchorPose), w, h)
            val p1 = projectPoint(localToWorld(base[i1][0], y0, base[i1][2], anchorPose), w, h)
            val p2 = projectPoint(localToWorld(top[i1][0], y1, top[i1][2], anchorPose), w, h)
            val p3 = projectPoint(localToWorld(top[i0][0], y1, top[i0][2], anchorPose), w, h)

            if (p0 != null && p1 != null && p2 != null && p3 != null) {
                perforatedMeshPaint.shader = getMeshShaderForColor(color)
                scratchPath.reset()
                scratchPath.moveTo(p0.x, p0.y)
                scratchPath.lineTo(p1.x, p1.y)
                scratchPath.lineTo(p2.x, p2.y)
                scratchPath.lineTo(p3.x, p3.y)
                scratchPath.close()
                canvas.drawPath(scratchPath, perforatedMeshPaint)

                wirePaint.color = color
                wirePaint.strokeWidth = if (isActive) 3.5f else 2.2f
                canvas.drawLine(p0.x, p0.y, p1.x, p1.y, wirePaint)
                canvas.drawLine(p1.x, p1.y, p2.x, p2.y, wirePaint)
                canvas.drawLine(p2.x, p2.y, p3.x, p3.y, wirePaint)
                canvas.drawLine(p3.x, p3.y, p0.x, p0.y, wirePaint)
            }
        }

        private fun drawFilletedCornerSector(
            canvas: Canvas, corner: Int,
            base: List<FloatArray>, top: List<FloatArray>,
            y0: Float, y1: Float,
            anchorPose: Pose,
            w: Float, h: Float, color: Int, isActive: Boolean
        ) {
            val startIdx = corner * 6 + 1
            perforatedMeshPaint.shader = getMeshShaderForColor(color)

            wirePaint.color = color
            wirePaint.strokeWidth = if (isActive) 3.0f else 2.0f

            for (step in 0 until 5) {
                val iA = (startIdx + step) % base.size
                val iB = (startIdx + step + 1) % base.size

                val p0 = projectPoint(localToWorld(base[iA][0], y0, base[iA][2], anchorPose), w, h)
                val p1 = projectPoint(localToWorld(base[iB][0], y0, base[iB][2], anchorPose), w, h)
                val p2 = projectPoint(localToWorld(top[iB][0], y1, top[iB][2], anchorPose), w, h)
                val p3 = projectPoint(localToWorld(top[iA][0], y1, top[iA][2], anchorPose), w, h)

                if (p0 != null && p1 != null && p2 != null && p3 != null) {
                    scratchPath.reset()
                    scratchPath.moveTo(p0.x, p0.y)
                    scratchPath.lineTo(p1.x, p1.y)
                    scratchPath.lineTo(p2.x, p2.y)
                    scratchPath.lineTo(p3.x, p3.y)
                    scratchPath.close()
                    canvas.drawPath(scratchPath, perforatedMeshPaint)

                    canvas.drawLine(p0.x, p0.y, p1.x, p1.y, wirePaint)
                    canvas.drawLine(p2.x, p2.y, p3.x, p3.y, wirePaint)
                    if (step == 0) canvas.drawLine(p0.x, p0.y, p3.x, p3.y, wirePaint)
                    if (step == 4) canvas.drawLine(p1.x, p1.y, p2.x, p2.y, wirePaint)
                }
            }
        }

        private fun drawTopCap(
            canvas: Canvas, top: List<FloatArray>, y: Float,
            anchorPose: Pose,
            w: Float, h: Float, color: Int, fill: Boolean
        ) {
            val pts = top.map { projectPoint(localToWorld(it[0], y, it[2], anchorPose), w, h) }
            if (pts.all { it != null }) {
                val isTopCaptured = capturedThumbnails.containsKey(8)
                val isTopActiveStep = currentStepIdx == 8
                val isActive = isTopCaptured || isTopActiveStep

                if (fill) {
                    perforatedMeshPaint.shader = getMeshShaderForColor(color)
                    scratchPath.reset()
                    scratchPath.moveTo(pts[0]!!.x, pts[0]!!.y)
                    for (i in 1 until pts.size) scratchPath.lineTo(pts[i]!!.x, pts[i]!!.y)
                    scratchPath.close()
                    canvas.drawPath(scratchPath, perforatedMeshPaint)
                }

                wirePaint.color = color
                wirePaint.strokeWidth = if (isActive) 3.5f else 2.5f
                for (i in pts.indices) {
                    val next = (i + 1) % pts.size
                    canvas.drawLine(pts[i]!!.x, pts[i]!!.y, pts[next]!!.x, pts[next]!!.y, wirePaint)
                }
            }
        }

        private fun drawPlanar3DReticle(canvas: Canvas, desc: FacetDesc, anchorPose: Pose, w: Float, h: Float) {
            val rColor = if (isAligned) Color.parseColor("#3FB950") else Color.parseColor("#F85149")
            reticlePaint.color = rColor
            fillReticlePaint.color = rColor

            val nx = desc.normX; val ny = desc.normY; val nz = desc.normZ
            val ux: Float; val uy: Float; val uz: Float

            if (abs(ny) > 0.95f) {
                ux = 1f; uy = 0f; uz = 0f
            } else {
                val len = sqrt(nz * nz + nx * nx)
                ux = nz / len; uy = 0f; uz = -nx / len
            }
            val vx = ny * uz - nz * uy
            val vy = nz * ux - nx * uz
            val vz = nx * uy - ny * ux

            val radius = min(prismWidth, prismDepth) * 0.20f

            val ringPts = mutableListOf<PointF?>()
            for (i in 0 until 16) {
                val angle = (2.0 * Math.PI * i / 16).toFloat()
                val px = desc.centerX + radius * (cos(angle) * ux + sin(angle) * vx) + nx * 0.003f
                val py = desc.centerY + radius * (cos(angle) * uy + sin(angle) * vy) + ny * 0.003f
                val pz = desc.centerZ + radius * (cos(angle) * uz + sin(angle) * vz) + nz * 0.003f
                ringPts.add(projectPoint(localToWorld(px, py, pz, anchorPose), w, h))
            }

            for (i in 0 until 16) {
                val next = (i + 1) % 16
                if (ringPts[i] != null && ringPts[next] != null) {
                    canvas.drawLine(ringPts[i]!!.x, ringPts[i]!!.y, ringPts[next]!!.x, ringPts[next]!!.y, reticlePaint)
                }
            }

            val ch = radius * 1.4f
            val h1 = projectPoint(localToWorld(desc.centerX - ux * ch + nx * 0.003f, desc.centerY - uy * ch + ny * 0.003f, desc.centerZ - uz * ch + nz * 0.003f, anchorPose), w, h)
            val h2 = projectPoint(localToWorld(desc.centerX + ux * ch + nx * 0.003f, desc.centerY + uy * ch + ny * 0.003f, desc.centerZ + uz * ch + nz * 0.003f, anchorPose), w, h)
            if (h1 != null && h2 != null) canvas.drawLine(h1.x, h1.y, h2.x, h2.y, reticlePaint)

            val centerPt = projectPoint(localToWorld(desc.centerX + nx * 0.003f, desc.centerY + ny * 0.003f, desc.centerZ + nz * 0.003f, anchorPose), w, h)
            if (centerPt != null) canvas.drawCircle(centerPt.x, centerPt.y, 14f, fillReticlePaint)
        }

        private fun projectPoint(x: Float, y: Float, z: Float, screenW: Float, screenH: Float): PointF? {
            scratchWorldVec[0] = x
            scratchWorldVec[1] = y
            scratchWorldVec[2] = z
            scratchWorldVec[3] = 1.0f
            Matrix.multiplyMV(scratchClipVec, 0, vpMatrix, 0, scratchWorldVec, 0)
            val w = scratchClipVec[3]
            if (w <= 0.01f) return null
            val ndcX = scratchClipVec[0] / w
            val ndcY = scratchClipVec[1] / w
            val sx = (ndcX * 0.5f + 0.5f) * screenW
            val sy = (-ndcY * 0.5f + 0.5f) * screenH
            return PointF(sx, sy)
        }

        private fun projectPoint(worldCoords: FloatArray, screenW: Float, screenH: Float): PointF? {
            return projectPoint(worldCoords[0], worldCoords[1], worldCoords[2], screenW, screenH)
        }
    }
}
