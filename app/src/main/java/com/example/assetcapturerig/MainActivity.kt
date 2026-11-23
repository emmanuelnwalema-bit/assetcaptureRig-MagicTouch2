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
import androidx.compose.ui.graphics.Color as ComposeColor
import com.google.android.filament.Box
import com.google.android.filament.Engine
import com.google.android.filament.IndexBuffer
import com.google.android.filament.MaterialInstance
import com.google.android.filament.RenderableManager
import com.google.android.filament.VertexBuffer
import com.google.ar.core.Anchor
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.math.Position
import io.github.sceneview.math.Rotation
import io.github.sceneview.math.Scale
import io.github.sceneview.node.MeshNode
import io.github.sceneview.node.Node
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
    private lateinit var overlayView: CrosshairOverlayView
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

    private val debugLog = StringBuilder("App Initialized (GPU AnchorNode + MagicTouch)\n")
    private var toneGen: ToneGenerator? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    private var tfliteInterpreter: Interpreter? = null
    private val modelAssetName = "magic_touch.tflite"

    private var currentStepIdx = 0
    private var isBoxPlaced = false
    private var isCapturing = false
    private var isBatchUploading = false
    private var isTakingCleanSnapshot = false
    private var lastFrame: Frame? = null
    private var prismAnchor: Anchor? = null
    private var trackedTablePlane: Plane? = null
    private var initialAzimuth = 0f
    private var alignStartTime: Long? = null
    private var cooldownUntil: Long = 0L
    private var lastHudUpdateMs: Long = 0L

    private var prismWidth = 0.10f
    private var prismDepth = 0.10f
    private var prismHeight = 0.05f

    // =========================================================
    // GPU SCENEGRAPH NODES & FILAMENT MATERIALS (ZERO PHASE LAG)
    // =========================================================
    private var gpuMaterialsReady = false
    private lateinit var matRedFace: MaterialInstance
    private lateinit var matGreenFace: MaterialInstance
    private lateinit var matGreyFace: MaterialInstance
    private lateinit var matRedWire: MaterialInstance
    private lateinit var matGreenWire: MaterialInstance
    private lateinit var matGreyWire: MaterialInstance
    private lateinit var matRingClingWall: MaterialInstance
    private lateinit var matRingClingWire: MaterialInstance
    private lateinit var matRingScanWall: MaterialInstance
    private lateinit var matRingScanWire: MaterialInstance

    // Pre-Lock GPU 3D Volumetric Encapsulating Ring
    private var liveRingRoot: Node? = null
    private var liveRingWallNode: MeshNode? = null
    private var liveRingWireNode: MeshNode? = null
    private var liveRingBuffers = mutableListOf<Pair<VertexBuffer, IndexBuffer>>()
    private var liveRingIsClingColor = false

    // Locked GPU 3D Bounding Prism attached to ARCore AnchorNode
    private var prismAnchorNode: AnchorNode? = null
    private var prismYawNode: Node? = null
    private val facetFaceNodes = arrayOfNulls<MeshNode>(9)
    private val facetWireNodes = arrayOfNulls<MeshNode>(9)
    private var structuralFaceNode: MeshNode? = null
    private var structuralWireNode: MeshNode? = null
    private var activeReticleNode: MeshNode? = null
    private var reticleBuiltForStep = -1
    private var lastReticleAlignedState = false
    private var lastVisualStepIdx = -1
    private val prismBuffers = mutableListOf<Pair<VertexBuffer, IndexBuffer>>()
    private var reticleBuffers: Pair<VertexBuffer, IndexBuffer>? = null

    // =========================================================
    // 3D ENCAPSULATING RING LIVE CLING STATE (PRE-LOCK)
    // =========================================================
    private var hasLiveCling = false
    private var liveClingConfidence = 0
    private var liveWorldX = 0f
    private var liveWorldY = 0f
    private var liveWorldZ = 0f
    private var liveWidth = 0.10f
    private var liveDepth = 0.10f
    private var liveHeight = 0.045f
    private var liveAzimuth = 0f
    private var liveCos2A = 1f
    private var liveSin2A = 0f
    private var liveClingScreenX = 0f
    private var liveClingScreenY = 0f
    private var frameCounter = 0

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

        // Keep the status banner lifted above the center aiming zone
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

        initMagicTouchTFLite()

        // Lightweight static 2D screen crosshair only (all 3D geometry is now GPU-rendered)
        overlayView = CrosshairOverlayView(this)
        rootLayout.addView(overlayView, 1)

        buildThumbnailStrip()

        sceneView.sessionConfiguration = { _, config ->
            config.focusMode = Config.FocusMode.AUTO
            config.planeFindingMode = Config.PlaneFindingMode.HORIZONTAL
            config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
        }

        sceneView.planeRenderer.isEnabled = false
        sceneView.planeRenderer.isVisible = false

        // Initialize Filament GPU materials and unit pre-lock 3D ring on the main thread
        sceneView.post {
            initGpuMaterialsAndLiveRing()
        }

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
            syncPrismYawTransform()
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
            syncPrismYawTransform()
            rebuildPrismGpuMeshes()
            Toast.makeText(this, "Rotated 90°", Toast.LENGTH_SHORT).show()
            true
        }

        updateChecklistUI()
        updateScaleLabels()
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            clearPrismGpuMeshes()
            destroyLiveRingGpuMeshes()
            prismAnchorNode?.let {
                sceneView.removeChildNode(it)
                it.destroy()
            }
            prismAnchor?.detach()
            tfliteInterpreter?.close()
        } catch (e: Exception) {}
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
                .setTitle("GPU AnchorNode + MagicTouch Diagnostics")
                .setView(sv)
                .setPositiveButton("OK", null)
                .setNeutralButton("Clear") { _, _ -> debugLog.clear() }
                .show()
        }
    }

    // =========================================================================
    // FILAMENT GPU MATERIALS & PROCEDURAL 3D MESH ENGINE
    // =========================================================================
    private fun initGpuMaterialsAndLiveRing() {
        if (gpuMaterialsReady) return
        try {
            val ml = sceneView.materialLoader

            // Translucent face fills (alpha < 1.0 selects Filament transparent material)
            matRedFace = ml.createColorInstance(
                color = ComposeColor(0.97f, 0.32f, 0.29f, 0.34f),
                metallic = 0.0f, roughness = 0.90f, reflectance = 0.10f
            )
            matGreenFace = ml.createColorInstance(
                color = ComposeColor(0.25f, 0.73f, 0.31f, 0.34f),
                metallic = 0.0f, roughness = 0.90f, reflectance = 0.10f
            )
            matGreyFace = ml.createColorInstance(
                color = ComposeColor(0.55f, 0.58f, 0.62f, 0.22f),
                metallic = 0.0f, roughness = 0.90f, reflectance = 0.10f
            )

            // Crisp opaque wireframe & reticle beam materials
            matRedWire = ml.createColorInstance(
                color = ComposeColor(0.99f, 0.30f, 0.27f, 1.0f),
                metallic = 0.0f, roughness = 0.65f, reflectance = 0.20f
            )
            matGreenWire = ml.createColorInstance(
                color = ComposeColor(0.22f, 0.85f, 0.34f, 1.0f),
                metallic = 0.0f, roughness = 0.65f, reflectance = 0.20f
            )
            matGreyWire = ml.createColorInstance(
                color = ComposeColor(0.62f, 0.66f, 0.70f, 1.0f),
                metallic = 0.0f, roughness = 0.75f, reflectance = 0.15f
            )

            // Pre-lock 3D Volumetric Encapsulating Ring materials
            matRingClingWall = ml.createColorInstance(
                color = ComposeColor(0.25f, 0.73f, 0.31f, 0.22f),
                metallic = 0.0f, roughness = 0.90f, reflectance = 0.10f
            )
            matRingClingWire = ml.createColorInstance(
                color = ComposeColor(0.25f, 0.88f, 0.36f, 1.0f),
                metallic = 0.0f, roughness = 0.60f, reflectance = 0.20f
            )
            matRingScanWall = ml.createColorInstance(
                color = ComposeColor(0.35f, 0.65f, 1.00f, 0.18f),
                metallic = 0.0f, roughness = 0.90f, reflectance = 0.10f
            )
            matRingScanWire = ml.createColorInstance(
                color = ComposeColor(0.35f, 0.65f, 1.00f, 1.0f),
                metallic = 0.0f, roughness = 0.60f, reflectance = 0.20f
            )

            buildUnitLiveRingGpuMesh()
            gpuMaterialsReady = true
            logDebug("✓ Filament GPU Materials & 3D Ring Ready")
        } catch (e: Exception) {
            logDebug("❌ GPU Material Init Failed: ${e.javaClass.simpleName}: ${e.message}")
        }
    }

    /**
     * Builds the pre-lock 3D Volumetric Encapsulating Ring once at unit scale (rx=1, rz=1, h=1).
     * Per-frame updates only modify liveRingRoot's position/rotation/scale (zero allocations).
     */
    private fun buildUnitLiveRingGpuMesh() {
        val engine = sceneView.engine
        destroyLiveRingGpuMeshes()

        val steps = 36
        val wallBuilder = GpuMeshBuilder()
        val wireBuilder = GpuMeshBuilder()

        val basePts = Array(steps) { FloatArray(3) }
        val midPts = Array(steps) { FloatArray(3) }
        val topPts = Array(steps) { FloatArray(3) }

        for (i in 0 until steps) {
            val ang = 2.0 * Math.PI * i / steps
            val c = cos(ang)
            val s = sin(ang)
            val lx = (c.sign * abs(c).pow(0.82)).toFloat()
            val lz = (s.sign * abs(s).pow(0.82)).toFloat()
            basePts[i] = floatArrayOf(lx, 0f, lz)
            midPts[i] = floatArrayOf(lx, 0.5f, lz)
            topPts[i] = floatArrayOf(lx, 1.0f, lz)
        }

        val barThick = 0.026f
        for (i in 0 until steps) {
            val next = (i + 1) % steps
            wallBuilder.addDoubleSidedQuad(basePts[i], basePts[next], topPts[next], topPts[i])

            wireBuilder.addCrossedBeam(basePts[i], basePts[next], barThick)
            wireBuilder.addCrossedBeam(topPts[i], topPts[next], barThick * 0.85f)
            if (i % 2 == 0) {
                wireBuilder.addCrossedBeam(midPts[i], midPts[next], barThick * 0.60f)
            }
            if (i % 3 == 0) {
                val ribThick = if (i % 9 == 0) barThick * 0.95f else barThick * 0.55f
                wireBuilder.addCrossedBeam(basePts[i], topPts[i], ribThick)
            }
        }

        val root = Node(engine).apply {
            isVisible = false
        }
        val (wallNode, wallBufs) = wallBuilder.buildMeshNode(engine, matRingScanWall)
        val (wireNode, wireBufs) = wireBuilder.buildMeshNode(engine, matRingScanWire)

        root.addChildNode(wallNode)
        root.addChildNode(wireNode)
        sceneView.addChildNode(root)

        liveRingRoot = root
        liveRingWallNode = wallNode
        liveRingWireNode = wireNode
        liveRingBuffers.add(wallBufs)
        liveRingBuffers.add(wireBufs)
        liveRingIsClingColor = false
    }

    private fun destroyLiveRingGpuMeshes() {
        val engine = sceneView.engine
        liveRingWallNode?.destroy()
        liveRingWireNode?.destroy()
        liveRingRoot?.let {
            sceneView.removeChildNode(it)
            it.destroy()
        }
        for ((vb, ib) in liveRingBuffers) {
            try { engine.destroyVertexBuffer(vb) } catch (_: Exception) {}
            try { engine.destroyIndexBuffer(ib) } catch (_: Exception) {}
        }
        liveRingBuffers.clear()
        liveRingRoot = null
        liveRingWallNode = null
        liveRingWireNode = null
    }

    private fun setNodeMaterial(node: Node?, mat: MaterialInstance) {
        val entity = node?.entity ?: return
        val rm = sceneView.engine.renderableManager
        val inst = rm.getInstance(entity)
        if (inst != 0) {
            rm.setMaterialInstanceAt(inst, 0, mat)
        }
    }

    private fun syncPrismYawTransform() {
        val yawDeg = Math.toDegrees(initialAzimuth.toDouble()).toFloat()
        prismYawNode?.rotation = Rotation(0f, yawDeg, 0f)
    }

    /**
     * Attaches (or re-attaches) the GPU 3D Bounding Prism to a plane-grounded ARCore AnchorNode
     * so Filament renders camera feed + 3D cage in the exact same GPU frame (0 ms lag).
     */
    private fun attachPrismToAnchor(newAnchor: Anchor) {
        val engine = sceneView.engine
        clearPrismGpuMeshes()

        prismAnchorNode?.let { oldNode ->
            sceneView.removeChildNode(oldNode)
            try { oldNode.destroy() } catch (_: Exception) {}
        }
        if (prismAnchor != null && prismAnchor != newAnchor) {
            try { prismAnchor?.detach() } catch (_: Exception) {}
        }

        prismAnchor = newAnchor

        val anchorNode = AnchorNode(engine = engine, anchor = newAnchor)
        val yawNode = Node(engine).apply {
            rotation = Rotation(0f, Math.toDegrees(initialAzimuth.toDouble()).toFloat(), 0f)
        }
        anchorNode.addChildNode(yawNode)
        sceneView.addChildNode(anchorNode)

        prismAnchorNode = anchorNode
        prismYawNode = yawNode

        rebuildPrismGpuMeshes()
    }

    /**
     * Rebuilds the 9 capturable facets + structural grey frame + active 3D planar reticle
     * on the GPU whenever prism dimensions change.
     */
    private fun rebuildPrismGpuMeshes() {
        if (!gpuMaterialsReady) return
        val yawNode = prismYawNode ?: return
        val engine = sceneView.engine

        clearPrismGpuMeshes()

        val hW = prismWidth / 2f
        val hD = prismDepth / 2f
        val r = min(hW, hD) * 0.36f
        val hMid = prismHeight * 0.50f
        val tan30 = tan(Math.toRadians(30.0)).toFloat()
        val dChamfer = min((prismHeight - hMid) * tan30, r * 0.82f)
        val rTop = max(r - dChamfer * 0.45f, 0.015f)

        val baseLoop = generateFilletedSquircleLoop(hW, hD, r)
        val midLoop = generateFilletedSquircleLoop(hW, hD, r)
        val topLoop = generateFilletedSquircleLoop(hW - dChamfer, hD - dChamfer, rTop)

        val wireThickActive = 0.0016f
        val wireThickStruct = 0.0011f

        val structFaceBuilder = GpuMeshBuilder()
        val structWireBuilder = GpuMeshBuilder()

        // Lower tier (y = 0..hMid): 4 straight walls (facets 0, 2, 4, 6) + 4 structural corners
        for (sector in 0 until 4) {
            val facetIdx = sector * 2
            val faceB = GpuMeshBuilder()
            val wireB = GpuMeshBuilder()

            val i0 = sector * 6
            val i1 = (i0 + 1) % baseLoop.size
            val p0 = floatArrayOf(baseLoop[i0][0], 0f, baseLoop[i0][2])
            val p1 = floatArrayOf(baseLoop[i1][0], 0f, baseLoop[i1][2])
            val p2 = floatArrayOf(midLoop[i1][0], hMid, midLoop[i1][2])
            val p3 = floatArrayOf(midLoop[i0][0], hMid, midLoop[i0][2])

            faceB.addDoubleSidedQuad(p0, p1, p2, p3)
            wireB.addCrossedBeam(p0, p1, wireThickActive)
            wireB.addCrossedBeam(p1, p2, wireThickActive)
            wireB.addCrossedBeam(p2, p3, wireThickActive)
            wireB.addCrossedBeam(p3, p0, wireThickActive)

            val (fNode, fBufs) = faceB.buildMeshNode(engine, matRedFace)
            val (wNode, wBufs) = wireB.buildMeshNode(engine, matRedWire)
            yawNode.addChildNode(fNode)
            yawNode.addChildNode(wNode)
            facetFaceNodes[facetIdx] = fNode
            facetWireNodes[facetIdx] = wNode
            prismBuffers.add(fBufs)
            prismBuffers.add(wBufs)

            // Structural lower filleted corner
            val startIdx = sector * 6 + 1
            for (step in 0 until 5) {
                val iA = (startIdx + step) % baseLoop.size
                val iB = (startIdx + step + 1) % baseLoop.size
                val c0 = floatArrayOf(baseLoop[iA][0], 0f, baseLoop[iA][2])
                val c1 = floatArrayOf(baseLoop[iB][0], 0f, baseLoop[iB][2])
                val c2 = floatArrayOf(midLoop[iB][0], hMid, midLoop[iB][2])
                val c3 = floatArrayOf(midLoop[iA][0], hMid, midLoop[iA][2])
                structFaceBuilder.addDoubleSidedQuad(c0, c1, c2, c3)
                structWireBuilder.addCrossedBeam(c0, c1, wireThickStruct)
                structWireBuilder.addCrossedBeam(c2, c3, wireThickStruct)
            }
        }

        // Upper chamfer tier (y = hMid..prismHeight): 4 structural beveled walls + 4 chamfer corners (facets 1, 3, 5, 7)
        for (sector in 0 until 4) {
            val i0 = sector * 6
            val i1 = (i0 + 1) % midLoop.size
            val s0 = floatArrayOf(midLoop[i0][0], hMid, midLoop[i0][2])
            val s1 = floatArrayOf(midLoop[i1][0], hMid, midLoop[i1][2])
            val s2 = floatArrayOf(topLoop[i1][0], prismHeight, topLoop[i1][2])
            val s3 = floatArrayOf(topLoop[i0][0], prismHeight, topLoop[i0][2])
            structFaceBuilder.addDoubleSidedQuad(s0, s1, s2, s3)
            structWireBuilder.addCrossedBeam(s0, s1, wireThickStruct)
            structWireBuilder.addCrossedBeam(s2, s3, wireThickStruct)
            structWireBuilder.addCrossedBeam(s0, s3, wireThickStruct)
            structWireBuilder.addCrossedBeam(s1, s2, wireThickStruct)

            val facetIdx = sector * 2 + 1
            val faceB = GpuMeshBuilder()
            val wireB = GpuMeshBuilder()
            val startIdx = sector * 6 + 1

            for (step in 0 until 5) {
                val iA = (startIdx + step) % midLoop.size
                val iB = (startIdx + step + 1) % midLoop.size
                val c0 = floatArrayOf(midLoop[iA][0], hMid, midLoop[iA][2])
                val c1 = floatArrayOf(midLoop[iB][0], hMid, midLoop[iB][2])
                val c2 = floatArrayOf(topLoop[iB][0], prismHeight, topLoop[iB][2])
                val c3 = floatArrayOf(topLoop[iA][0], prismHeight, topLoop[iA][2])
                faceB.addDoubleSidedQuad(c0, c1, c2, c3)
                wireB.addCrossedBeam(c0, c1, wireThickActive)
                wireB.addCrossedBeam(c2, c3, wireThickActive)
                if (step == 0) wireB.addCrossedBeam(c0, c3, wireThickActive)
                if (step == 4) wireB.addCrossedBeam(c1, c2, wireThickActive)
            }

            val (fNode, fBufs) = faceB.buildMeshNode(engine, matRedFace)
            val (wNode, wBufs) = wireB.buildMeshNode(engine, matRedWire)
            yawNode.addChildNode(fNode)
            yawNode.addChildNode(wNode)
            facetFaceNodes[facetIdx] = fNode
            facetWireNodes[facetIdx] = wNode
            prismBuffers.add(fBufs)
            prismBuffers.add(wBufs)
        }

        // Top Cap (facet 8)
        val topFaceB = GpuMeshBuilder()
        val topWireB = GpuMeshBuilder()
        val topRing3D = topLoop.map { floatArrayOf(it[0], prismHeight, it[2]) }
        topFaceB.addDoubleSidedConvexPolygon(topRing3D)
        for (i in topRing3D.indices) {
            val next = (i + 1) % topRing3D.size
            topWireB.addCrossedBeam(topRing3D[i], topRing3D[next], wireThickActive)
        }
        val (topFNode, topFBufs) = topFaceB.buildMeshNode(engine, matRedFace)
        val (topWNode, topWBufs) = topWireB.buildMeshNode(engine, matRedWire)
        yawNode.addChildNode(topFNode)
        yawNode.addChildNode(topWNode)
        facetFaceNodes[8] = topFNode
        facetWireNodes[8] = topWNode
        prismBuffers.add(topFBufs)
        prismBuffers.add(topWBufs)

        // Structural combined grey node
        val (stFNode, stFBufs) = structFaceBuilder.buildMeshNode(engine, matGreyFace)
        val (stWNode, stWBufs) = structWireBuilder.buildMeshNode(engine, matGreyWire)
        yawNode.addChildNode(stFNode)
        yawNode.addChildNode(stWNode)
        structuralFaceNode = stFNode
        structuralWireNode = stWNode
        prismBuffers.add(stFBufs)
        prismBuffers.add(stWBufs)

        reticleBuiltForStep = -1
        lastVisualStepIdx = -1
        updateGpuFacetMaterialsAndReticle(isAlignedNow = false, forceUpdate = true)
    }

    private fun clearPrismGpuMeshes() {
        val engine = sceneView.engine
        val yawNode = prismYawNode

        for (i in 0 until 9) {
            facetFaceNodes[i]?.let { yawNode?.removeChildNode(it); try { it.destroy() } catch (_: Exception) {} }
            facetWireNodes[i]?.let { yawNode?.removeChildNode(it); try { it.destroy() } catch (_: Exception) {} }
            facetFaceNodes[i] = null
            facetWireNodes[i] = null
        }
        structuralFaceNode?.let { yawNode?.removeChildNode(it); try { it.destroy() } catch (_: Exception) {} }
        structuralWireNode?.let { yawNode?.removeChildNode(it); try { it.destroy() } catch (_: Exception) {} }
        structuralFaceNode = null
        structuralWireNode = null

        clearReticleGpuNode()

        for ((vb, ib) in prismBuffers) {
            try { engine.destroyVertexBuffer(vb) } catch (_: Exception) {}
            try { engine.destroyIndexBuffer(ib) } catch (_: Exception) {}
        }
        prismBuffers.clear()
    }

    private fun clearReticleGpuNode() {
        val engine = sceneView.engine
        activeReticleNode?.let {
            prismYawNode?.removeChildNode(it)
            try { it.destroy() } catch (_: Exception) {}
        }
        activeReticleNode = null
        reticleBuffers?.let { (vb, ib) ->
            try { engine.destroyVertexBuffer(vb) } catch (_: Exception) {}
            try { engine.destroyIndexBuffer(ib) } catch (_: Exception) {}
        }
        reticleBuffers = null
        reticleBuiltForStep = -1
    }

    /**
     * O(1) GPU material state update for all 9 facets + active 3D planar bullseye reticle.
     */
    private fun updateGpuFacetMaterialsAndReticle(isAlignedNow: Boolean, forceUpdate: Boolean = false) {
        if (!gpuMaterialsReady || prismYawNode == null) return
        if (!forceUpdate && currentStepIdx == lastVisualStepIdx && isAlignedNow == lastReticleAlignedState) return

        for (i in 0 until 9) {
            val isCaptured = capturedThumbnails.containsKey(i)
            val isActive = (currentStepIdx == i)
            val useGreen = isCaptured || (isActive && isAlignedNow)
            setNodeMaterial(facetFaceNodes[i], if (useGreen) matGreenFace else matRedFace)
            setNodeMaterial(facetWireNodes[i], if (useGreen) matGreenWire else matRedWire)
        }

        if (currentStepIdx >= sequence.size) {
            clearReticleGpuNode()
        } else {
            if (reticleBuiltForStep != currentStepIdx) {
                rebuildActiveReticleMesh(currentStepIdx)
            }
            setNodeMaterial(activeReticleNode, if (isAlignedNow) matGreenWire else matRedWire)
        }

        lastVisualStepIdx = currentStepIdx
        lastReticleAlignedState = isAlignedNow
    }

    private fun rebuildActiveReticleMesh(step: Int) {
        val yawNode = prismYawNode ?: return
        val engine = sceneView.engine
        clearReticleGpuNode()

        val desc = getFacetDescriptor(step)
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

        val radius = min(prismWidth, prismDepth) * 0.21f
        val lift = 0.0035f
        val cx = desc.centerX + nx * lift
        val cy = desc.centerY + ny * lift
        val cz = desc.centerZ + nz * lift

        val builder = GpuMeshBuilder()
        val ringSteps = 20
        val ringPts = Array(ringSteps) { i ->
            val ang = (2.0 * Math.PI * i / ringSteps).toFloat()
            val c = cos(ang); val s = sin(ang)
            floatArrayOf(
                cx + radius * (c * ux + s * vx),
                cy + radius * (c * uy + s * vy),
                cz + radius * (c * uz + s * vz)
            )
        }

        val beamThick = 0.0018f
        for (i in 0 until ringSteps) {
            val next = (i + 1) % ringSteps
            builder.addCrossedBeam(ringPts[i], ringPts[next], beamThick)
        }

        // Crosshair ticks
        val ch = radius * 1.40f
        builder.addCrossedBeam(
            floatArrayOf(cx - ux * ch, cy - uy * ch, cz - uz * ch),
            floatArrayOf(cx + ux * ch, cy + uy * ch, cz + uz * ch),
            beamThick
        )
        builder.addCrossedBeam(
            floatArrayOf(cx - vx * ch * 0.6f, cy - vy * ch * 0.6f, cz - vz * ch * 0.6f),
            floatArrayOf(cx + vx * ch * 0.6f, cy + vy * ch * 0.6f, cz + vz * ch * 0.6f),
            beamThick * 0.85f
        )

        // Center bullseye dot disc
        val dotR = (radius * 0.28f).coerceIn(0.003f, 0.009f)
        val dotPts = (0 until 12).map { i ->
            val ang = (2.0 * Math.PI * i / 12).toFloat()
            val c = cos(ang); val s = sin(ang)
            floatArrayOf(
                cx + dotR * (c * ux + s * vx),
                cy + dotR * (c * uy + s * vy),
                cz + dotR * (c * uz + s * vz)
            )
        }
        builder.addDoubleSidedConvexPolygon(dotPts)

        val (rNode, rBufs) = builder.buildMeshNode(engine, matRedWire)
        yawNode.addChildNode(rNode)
        activeReticleNode = rNode
        reticleBuffers = rBufs
        reticleBuiltForStep = step
    }

    /**
     * Helper that builds double-sided Filament VertexBuffer/IndexBuffer geometry
     * with valid POSITION (FLOAT3) and TANGENTS (FLOAT4 quaternion) attributes.
     */
    private class GpuMeshBuilder {
        private val positions = mutableListOf<Float>()
        private val tangents = mutableListOf<Float>()
        private val indices = mutableListOf<Short>()

        private fun normalToQuaternion(nx: Float, ny: Float, nz: Float): FloatArray {
            val len = sqrt(nx * nx + ny * ny + nz * nz)
            if (len < 1e-6f) return floatArrayOf(0f, 0f, 0f, 1f)
            val x = nx / len; val y = ny / len; val z = nz / len
            if (z < -0.999f) return floatArrayOf(0f, 1f, 0f, 0f)
            val qx = -y
            val qy = x
            val qz = 0f
            val qw = 1f + z
            val qLen = sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            return floatArrayOf(qx / qLen, qy / qLen, qz / qLen, qw / qLen)
        }

        private fun addVertex(p: FloatArray, q: FloatArray): Short {
            val idx = (positions.size / 3).toShort()
            positions.add(p[0]); positions.add(p[1]); positions.add(p[2])
            tangents.add(q[0]); tangents.add(q[1]); tangents.add(q[2]); tangents.add(q[3])
            return idx
        }

        fun addDoubleSidedQuad(p0: FloatArray, p1: FloatArray, p2: FloatArray, p3: FloatArray) {
            val e1x = p1[0] - p0[0]; val e1y = p1[1] - p0[1]; val e1z = p1[2] - p0[2]
            val e2x = p2[0] - p0[0]; val e2y = p2[1] - p0[1]; val e2z = p2[2] - p0[2]
            val nx = e1y * e2z - e1z * e2y
            val ny = e1z * e2x - e1x * e2z
            val nz = e1x * e2y - e1y * e2x

            val qF = normalToQuaternion(nx, ny, nz)
            val f0 = addVertex(p0, qF); val f1 = addVertex(p1, qF)
            val f2 = addVertex(p2, qF); val f3 = addVertex(p3, qF)
            indices.add(f0); indices.add(f1); indices.add(f2)
            indices.add(f0); indices.add(f2); indices.add(f3)

            val qB = normalToQuaternion(-nx, -ny, -nz)
            val b0 = addVertex(p0, qB); val b1 = addVertex(p1, qB)
            val b2 = addVertex(p2, qB); val b3 = addVertex(p3, qB)
            indices.add(b0); indices.add(b2); indices.add(b1)
            indices.add(b0); indices.add(b3); indices.add(b2)
        }

        fun addDoubleSidedConvexPolygon(pts: List<FloatArray>) {
            if (pts.size < 3) return
            var cx = 0f; var cy = 0f; var cz = 0f
            for (p in pts) { cx += p[0]; cy += p[1]; cz += p[2] }
            val center = floatArrayOf(cx / pts.size, cy / pts.size, cz / pts.size)

            val e1x = pts[1][0] - pts[0][0]; val e1y = pts[1][1] - pts[0][1]; val e1z = pts[1][2] - pts[0][2]
            val e2x = pts[2][0] - pts[0][0]; val e2y = pts[2][1] - pts[0][1]; val e2z = pts[2][2] - pts[0][2]
            val nx = e1y * e2z - e1z * e2y
            val ny = e1z * e2x - e1x * e2z
            val nz = e1x * e2y - e1y * e2x

            val qF = normalToQuaternion(nx, ny, nz)
            val qB = normalToQuaternion(-nx, -ny, -nz)

            val fCenter = addVertex(center, qF)
            val fRing = pts.map { addVertex(it, qF) }
            for (i in pts.indices) {
                val next = (i + 1) % pts.size
                indices.add(fCenter); indices.add(fRing[i]); indices.add(fRing[next])
            }

            val bCenter = addVertex(center, qB)
            val bRing = pts.map { addVertex(it, qB) }
            for (i in pts.indices) {
                val next = (i + 1) % pts.size
                indices.add(bCenter); indices.add(bRing[next]); indices.add(bRing[i])
            }
        }

        /**
         * Builds a crisp 3D crossed-ribbon beam between a and b so wireframe lines
         * remain solid and thick from every 3D viewing angle on mobile GPUs.
         */
        fun addCrossedBeam(a: FloatArray, b: FloatArray, halfThick: Float) {
            val dx = b[0] - a[0]; val dy = b[1] - a[1]; val dz = b[2] - a[2]
            val len = sqrt(dx * dx + dy * dy + dz * dz)
            if (len < 1e-5f) return
            val tx = dx / len; val ty = dy / len; val tz = dz / len

            // Pick a reference vector not parallel to tangent
            val rx: Float; val ry: Float; val rz: Float
            if (abs(ty) < 0.9f) {
                rx = 0f; ry = 1f; rz = 0f
            } else {
                rx = 1f; ry = 0f; rz = 0f
            }

            // First orthogonal axis u = normalize(t x r)
            var ux = ty * rz - tz * ry
            var uy = tz * rx - tx * rz
            var uz = tx * ry - ty * rx
            val uLen = sqrt(ux * ux + uy * uy + uz * uz).coerceAtLeast(1e-5f)
            ux = (ux / uLen) * halfThick; uy = (uy / uLen) * halfThick; uz = (uz / uLen) * halfThick

            // Second orthogonal axis v = normalize(t x u)
            var vx = ty * uz - tz * uy
            var vy = tz * ux - tx * uz
            var vz = tx * uy - ty * ux
            val vLen = sqrt(vx * vx + vy * vy + vz * vz).coerceAtLeast(1e-5f)
            vx = (vx / vLen) * halfThick; vy = (vy / vLen) * halfThick; vz = (vz / vLen) * halfThick

            // Ribbon 1 along u
            addDoubleSidedQuad(
                floatArrayOf(a[0] - ux, a[1] - uy, a[2] - uz),
                floatArrayOf(a[0] + ux, a[1] + uy, a[2] + uz),
                floatArrayOf(b[0] + ux, b[1] + uy, b[2] + uz),
                floatArrayOf(b[0] - ux, b[1] - uy, b[2] - uz)
            )
            // Ribbon 2 along v
            addDoubleSidedQuad(
                floatArrayOf(a[0] - vx, a[1] - vy, a[2] - vz),
                floatArrayOf(a[0] + vx, a[1] + vy, a[2] + vz),
                floatArrayOf(b[0] + vx, b[1] + vy, b[2] + vz),
                floatArrayOf(b[0] - vx, b[1] - vy, b[2] - vz)
            )
        }

        fun buildMeshNode(engine: Engine, material: MaterialInstance): Pair<MeshNode, Pair<VertexBuffer, IndexBuffer>> {
            val vCount = max(1, positions.size / 3)
            val iCount = max(3, indices.size)

            val posBuf = ByteBuffer.allocateDirect(vCount * 3 * 4).order(ByteOrder.nativeOrder())
            for (v in positions) posBuf.putFloat(v)
            while (posBuf.hasRemaining()) posBuf.putFloat(0f)
            posBuf.rewind()

            val tanBuf = ByteBuffer.allocateDirect(vCount * 4 * 4).order(ByteOrder.nativeOrder())
            for (v in tangents) tanBuf.putFloat(v)
            while (tanBuf.hasRemaining()) tanBuf.putFloat(1f)
            tanBuf.rewind()

            val idxBuf = ByteBuffer.allocateDirect(iCount * 2).order(ByteOrder.nativeOrder())
            for (idx in indices) idxBuf.putShort(idx)
            while (idxBuf.hasRemaining()) idxBuf.putShort(0)
            idxBuf.rewind()

            val vb = VertexBuffer.Builder()
                .bufferCount(2)
                .vertexCount(vCount)
                .attribute(VertexBuffer.VertexAttribute.POSITION, 0, VertexBuffer.AttributeType.FLOAT3)
                .attribute(VertexBuffer.VertexAttribute.TANGENTS, 1, VertexBuffer.AttributeType.FLOAT4)
                .build(engine)
            vb.setBufferAt(engine, 0, posBuf)
            vb.setBufferAt(engine, 1, tanBuf)

            val ib = IndexBuffer.Builder()
                .indexCount(iCount)
                .bufferType(IndexBuffer.Builder.IndexType.USHORT)
                .build(engine)
            ib.setBuffer(engine, idxBuf)

            val node = MeshNode(
                engine = engine,
                primitiveType = RenderableManager.PrimitiveType.TRIANGLES,
                vertexBuffer = vb,
                indexBuffer = ib,
                materialInstance = material,
                builder = {
                    boundingBox(Box(0f, 0.2f, 0f, 1.5f, 1.5f, 1.5f))
                    culling(false)
                    castShadows(false)
                    receiveShadows(false)
                }
            )
            return Pair(node, Pair(vb, ib))
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

    // =========================================================================
    // TFLITE MAGICTOUCH INITIALIZATION & THUMBNAIL STRIP
    // =========================================================================
    private fun initMagicTouchTFLite() {
        Thread {
            try {
                logDebug("Loading TFLite asset: $modelAssetName...")
                val fd: AssetFileDescriptor = assets.openFd(modelAssetName)
                val inputStream = FileInputStream(fd.fileDescriptor)
                val fileChannel = inputStream.channel
                val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)

                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                }
                val interp = Interpreter(modelBuffer, options)
                tfliteInterpreter = interp

                val inShape = interp.getInputTensor(0).shape()
                val outShape = interp.getOutputTensor(0).shape()
                logDebug("✓ MagicTouch Ready! In:${inShape.contentToString()} Out:${outShape.contentToString()}")
            } catch (e: Exception) {
                logDebug("❌ MagicTouch Load Failed: ${e.javaClass.simpleName}: ${e.message}")
            }
        }.start()
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
        rebuildPrismGpuMeshes()
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
    // PHASE 1: OUTSIDE-IN SILHOUETTE DETECTOR & GPU 3D ENCAPSULATING RING
    // =========================================================================
    private fun updateLiveClingReticle(frame: Frame, planeHitPose: Pose) {
        val sWidth = sceneView.width.toFloat()
        val sHeight = sceneView.height.toFloat()
        if (sWidth <= 0f || sHeight <= 0f) return

        var cameraImage: Image? = null
        try {
            cameraImage = frame.acquireCameraImage()
            val imgW = cameraImage.width
            val imgH = cameraImage.height

            val yPlane = cameraImage.planes[0]
            val uPlane = cameraImage.planes[1]
            val vPlane = cameraImage.planes[2]

            val yBuf = yPlane.buffer
            val uBuf = uPlane.buffer
            val vBuf = vPlane.buffer

            val yRowStride = yPlane.rowStride
            val yPixStride = yPlane.pixelStride
            val uvRowStride = uPlane.rowStride
            val uvPixStride = uPlane.pixelStride

            val minDim = min(sWidth, sHeight)
            val roiHalf = minDim * 0.34f

            val searchCX = if (hasLiveCling) {
                (sWidth * 0.5f) * 0.72f + liveClingScreenX * 0.28f
            } else {
                sWidth * 0.5f
            }
            val searchCY = if (hasLiveCling) {
                (sHeight * 0.5f) * 0.72f + liveClingScreenY * 0.28f
            } else {
                sHeight * 0.5f
            }

            val gridN = 31
            val numGrid = gridN * gridN
            val viewPts = FloatArray(numGrid * 2)
            val imgPts = FloatArray(numGrid * 2)

            var idx = 0
            for (gy in 0 until gridN) {
                val fy = (gy.toFloat() / (gridN - 1)) * 2f - 1f
                val sy = (searchCY + fy * roiHalf).coerceIn(4f, sHeight - 5f)
                for (gx in 0 until gridN) {
                    val fx = (gx.toFloat() / (gridN - 1)) * 2f - 1f
                    val sx = (searchCX + fx * roiHalf).coerceIn(4f, sWidth - 5f)
                    viewPts[idx++] = sx
                    viewPts[idx++] = sy
                }
            }

            frame.transformCoordinates2d(
                Coordinates2d.VIEW, viewPts,
                Coordinates2d.IMAGE_PIXELS, imgPts
            )

            val cellY = IntArray(numGrid)
            val cellU = IntArray(numGrid)
            val cellV = IntArray(numGrid)

            for (g in 0 until numGrid) {
                val ix = imgPts[g * 2].toInt().coerceIn(1, imgW - 2)
                val iy = imgPts[g * 2 + 1].toInt().coerceIn(1, imgH - 2)

                val y00 = yBuf.get(iy * yRowStride + ix * yPixStride).toInt() and 0xFF
                val y10 = yBuf.get(iy * yRowStride + (ix + 1) * yPixStride).toInt() and 0xFF
                val y01 = yBuf.get((iy + 1) * yRowStride + ix * yPixStride).toInt() and 0xFF
                val y11 = yBuf.get((iy + 1) * yRowStride + (ix + 1) * yPixStride).toInt() and 0xFF
                cellY[g] = (y00 + y10 + y01 + y11) shr 2

                val uvOff = (iy / 2) * uvRowStride + (ix / 2) * uvPixStride
                cellU[g] = uBuf.get(uvOff).toInt() and 0xFF
                cellV[g] = vBuf.get(uvOff).toInt() and 0xFF
            }

            var jxx = 0f; var jyy = 0f; var jxy = 0f
            for (gy in 1 until gridN - 1) {
                for (gx in 1 until gridN - 1) {
                    if (gx <= 2 || gx >= gridN - 3 || gy <= 2 || gy >= gridN - 3) {
                        val g = gy * gridN + gx
                        val dx = (cellY[g + 1] - cellY[g - 1]).toFloat()
                        val dy = (cellY[g + gridN] - cellY[g - gridN]).toFloat()
                        jxx += dx * dx
                        jyy += dy * dy
                        jxy += dx * dy
                    }
                }
            }

            val gradAngle = 0.5f * atan2(2f * jxy, jxx - jyy)
            val stripeAngle = gradAngle + (Math.PI / 2.0).toFloat()
            val sCos = cos(stripeAngle)
            val sSin = sin(stripeAngle)

            val rawFg = BooleanArray(numGrid)

            for (gy in 2 until gridN - 2) {
                for (gx in 2 until gridN - 2) {
                    val g = gy * gridN + gx
                    val py = cellY[g]
                    val pu = cellU[g]
                    val pv = cellV[g]

                    var tPos = 1f
                    var b1x = gx; var b1y = gy
                    while (tPos < gridN) {
                        val nx = (gx + tPos * sCos).roundToInt()
                        val ny = (gy + tPos * sSin).roundToInt()
                        if (nx <= 1 || nx >= gridN - 2 || ny <= 1 || ny >= gridN - 2) {
                            b1x = nx.coerceIn(1, gridN - 2)
                            b1y = ny.coerceIn(1, gridN - 2)
                            break
                        }
                        tPos += 1f
                    }

                    var tNeg = 1f
                    var b2x = gx; var b2y = gy
                    while (tNeg < gridN) {
                        val nx = (gx - tNeg * sCos).roundToInt()
                        val ny = (gy - tNeg * sSin).roundToInt()
                        if (nx <= 1 || nx >= gridN - 2 || ny <= 1 || ny >= gridN - 2) {
                            b2x = nx.coerceIn(1, gridN - 2)
                            b2y = ny.coerceIn(1, gridN - 2)
                            break
                        }
                        tNeg += 1f
                    }

                    var minStripeDiff = Int.MAX_VALUE
                    for (oy in -1..1) {
                        for (ox in -1..1) {
                            val idx1 = (b1y + oy).coerceIn(0, gridN - 1) * gridN + (b1x + ox).coerceIn(0, gridN - 1)
                            val idx2 = (b2y + oy).coerceIn(0, gridN - 1) * gridN + (b2x + ox).coerceIn(0, gridN - 1)

                            val predY = (cellY[idx1] + cellY[idx2]) shr 1
                            val predU = (cellU[idx1] + cellU[idx2]) shr 1
                            val predV = (cellV[idx1] + cellV[idx2]) shr 1

                            val dy = py - predY
                            val du = (pu - predU) * 2
                            val dv = (pv - predV) * 2
                            val isSoftShadow = (dy in -45..-8) && (du * du + dv * dv < 140)
                            val dSq = if (isSoftShadow) 0 else (dy * dy + du * du + dv * dv)
                            if (dSq < minStripeDiff) minStripeDiff = dSq

                            val dy1 = py - cellY[idx1]
                            val du1 = (pu - cellU[idx1]) * 2
                            val dv1 = (pv - cellV[idx1]) * 2
                            val dSq1 = dy1 * dy1 + du1 * du1 + dv1 * dv1
                            if (dSq1 < minStripeDiff) minStripeDiff = dSq1
                        }
                    }

                    val sxStep = sCos.roundToInt().coerceIn(-1, 1)
                    val syStep = sSin.roundToInt().coerceIn(-1, 1)
                    val gFwd = (gy + syStep) * gridN + (gx + sxStep)
                    val gBwd = (gy - syStep) * gridN + (gx - sxStep)
                    val alongStripeGrad = abs(cellY[gFwd] - cellY[gBwd]) +
                        (abs(cellU[gFwd] - cellU[gBwd]) + abs(cellV[gFwd] - cellV[gBwd])) * 2

                    rawFg[g] = (minStripeDiff > 320) || (alongStripeGrad > 36)
                }
            }

            val dilatedFg = BooleanArray(numGrid)
            for (gy in 2 until gridN - 2) {
                for (gx in 2 until gridN - 2) {
                    var countN = 0
                    for (ky in -1..1) {
                        for (kx in -1..1) {
                            if (rawFg[(gy + ky) * gridN + (gx + kx)]) countN++
                        }
                    }
                    dilatedFg[gy * gridN + gx] = countN >= 3
                }
            }

            val outsideBg = BooleanArray(numGrid)
            val bgQueue = ArrayDeque<Int>()
            for (gy in 0 until gridN) {
                for (gx in 0 until gridN) {
                    if (gx <= 1 || gx >= gridN - 2 || gy <= 1 || gy >= gridN - 2) {
                        val bIdx = gy * gridN + gx
                        outsideBg[bIdx] = true
                        bgQueue.add(bIdx)
                    }
                }
            }

            while (!bgQueue.isEmpty()) {
                val curr = bgQueue.removeFirst()
                val cx = curr % gridN
                val cy = curr / gridN
                val edges = intArrayOf(
                    cy * gridN + (cx - 1).coerceAtLeast(0),
                    cy * gridN + (cx + 1).coerceAtMost(gridN - 1),
                    (cy - 1).coerceAtLeast(0) * gridN + cx,
                    (cy + 1).coerceAtMost(gridN - 1) * gridN + cx
                )
                for (nIdx in edges) {
                    if (!outsideBg[nIdx] && !dilatedFg[nIdx]) {
                        outsideBg[nIdx] = true
                        bgQueue.add(nIdx)
                    }
                }
            }

            val solidSilhouette = BooleanArray(numGrid)
            for (gy in 2 until gridN - 2) {
                for (gx in 2 until gridN - 2) {
                    var nonBgNeighbors = 0
                    for (ky in -1..1) {
                        for (kx in -1..1) {
                            if (!outsideBg[(gy + ky) * gridN + (gx + kx)]) nonBgNeighbors++
                        }
                    }
                    solidSilhouette[gy * gridN + gx] = nonBgNeighbors >= 5
                }
            }

            val centerG = gridN / 2
            var seedIdx = -1
            var bestDist = Int.MAX_VALUE
            for (gy in 3 until gridN - 3) {
                for (gx in 3 until gridN - 3) {
                    val g = gy * gridN + gx
                    if (solidSilhouette[g]) {
                        val d = (gx - centerG) * (gx - centerG) + (gy - centerG) * (gy - centerG)
                        if (d < bestDist) {
                            bestDist = d
                            seedIdx = g
                        }
                    }
                }
            }

            if (seedIdx == -1 || bestDist > 45) {
                liveClingConfidence = (liveClingConfidence - 1).coerceAtLeast(0)
                if (liveClingConfidence == 0) hasLiveCling = false
                return
            }

            val compVisited = BooleanArray(numGrid)
            val compQueue = ArrayDeque<Int>()
            compQueue.add(seedIdx)
            compVisited[seedIdx] = true

            val compX = mutableListOf<Float>()
            val compY = mutableListOf<Float>()
            var sumX = 0f
            var sumY = 0f

            while (!compQueue.isEmpty()) {
                val curr = compQueue.removeFirst()
                val cx = curr % gridN
                val cy = curr / gridN

                val sx = viewPts[curr * 2]
                val sy = viewPts[curr * 2 + 1]
                compX.add(sx)
                compY.add(sy)
                sumX += sx
                sumY += sy

                for (ny in (cy - 1)..(cy + 1)) {
                    for (nx in (cx - 1)..(cx + 1)) {
                        if (nx in 2 until (gridN - 2) && ny in 2 until (gridN - 2)) {
                            val nIdx = ny * gridN + nx
                            if (solidSilhouette[nIdx] && !compVisited[nIdx]) {
                                compVisited[nIdx] = true
                                compQueue.add(nIdx)
                            }
                        }
                    }
                }
            }

            val count = compX.size
            if (count < 12 || count > (numGrid * 0.72f).toInt()) {
                liveClingConfidence = (liveClingConfidence - 1).coerceAtLeast(0)
                if (liveClingConfidence == 0) hasLiveCling = false
                return
            }

            val barX = sumX / count
            val barY = sumY / count

            var bestArea = Float.MAX_VALUE
            var bestRad = 0f
            for (deg in 0 until 90 step 3) {
                val rad = Math.toRadians(deg.toDouble()).toFloat()
                val c = cos(rad)
                val s = sin(rad)
                var minU = Float.MAX_VALUE; var maxU = -Float.MAX_VALUE
                var minV = Float.MAX_VALUE; var maxV = -Float.MAX_VALUE

                for (i in 0 until count) {
                    val dx = compX[i] - barX
                    val dy = compY[i] - barY
                    val pu = dx * c + dy * s
                    val pv = -dx * s + dy * c
                    if (pu < minU) minU = pu
                    if (pu > maxU) maxU = pu
                    if (pv < minV) minV = pv
                    if (pv > maxV) maxV = pv
                }
                val area = (maxU - minU).coerceAtLeast(12f) * (maxV - minV).coerceAtLeast(12f)
                if (area < bestArea) {
                    bestArea = area
                    bestRad = rad
                }
            }

            val uX = cos(bestRad)
            val uY = sin(bestRad)
            val vX = -uY
            val vY = uX

            var minU = Float.MAX_VALUE; var maxU = -Float.MAX_VALUE
            var minV = Float.MAX_VALUE; var maxV = -Float.MAX_VALUE
            for (i in 0 until count) {
                val dx = compX[i] - barX
                val dy = compY[i] - barY
                val pu = dx * uX + dy * uY
                val pv = dx * vX + dy * vY
                if (pu < minU) minU = pu
                if (pu > maxU) maxU = pu
                if (pv < minV) minV = pv
                if (pv > maxV) maxV = pv
            }

            val cellStepPx = (roiHalf * 2f) / (gridN - 1)
            val padPx = cellStepPx * 0.85f
            val midU = (minU + maxU) * 0.5f
            val midV = (minV + maxV) * 0.5f
            val halfSpan1Px = ((maxU - minU) * 0.5f + padPx).coerceAtLeast(18f)
            val halfSpan2Px = ((maxV - minV) * 0.5f + padPx).coerceAtLeast(18f)

            val boxScreenCX = barX + midU * uX + midV * vX
            val boxScreenCY = barY + midU * uY + midV * vY

            val vMat = FloatArray(16)
            val pMat = FloatArray(16)
            val vpMat = FloatArray(16)
            val invVpMat = FloatArray(16)

            frame.camera.getViewMatrix(vMat, 0)
            frame.camera.getProjectionMatrix(pMat, 0, 0.05f, 50.0f)
            Matrix.multiplyMM(vpMat, 0, pMat, 0, vMat, 0)
            Matrix.invertM(invVpMat, 0, vpMat, 0)

            val planeY = planeHitPose.ty()
            val estMidY = planeY + liveHeight * 0.5f

            val pCenterMid = intersectScreenRayWithHorizontalPlane(boxScreenCX, boxScreenCY, sWidth, sHeight, invVpMat, estMidY)
            val p1AMid = intersectScreenRayWithHorizontalPlane(boxScreenCX + halfSpan1Px * uX, boxScreenCY + halfSpan1Px * uY, sWidth, sHeight, invVpMat, estMidY)
            val p1BMid = intersectScreenRayWithHorizontalPlane(boxScreenCX - halfSpan1Px * uX, boxScreenCY - halfSpan1Px * uY, sWidth, sHeight, invVpMat, estMidY)
            val p2AMid = intersectScreenRayWithHorizontalPlane(boxScreenCX + halfSpan2Px * vX, boxScreenCY + halfSpan2Px * vY, sWidth, sHeight, invVpMat, estMidY)
            val p2BMid = intersectScreenRayWithHorizontalPlane(boxScreenCX - halfSpan2Px * vX, boxScreenCY - halfSpan2Px * vY, sWidth, sHeight, invVpMat, estMidY)

            if (pCenterMid == null || p1AMid == null || p1BMid == null || p2AMid == null || p2BMid == null) return

            val v1X = p1AMid[0] - p1BMid[0]
            val v1Z = p1AMid[2] - p1BMid[2]
            val span1 = sqrt(v1X * v1X + v1Z * v1Z).coerceIn(0.045f, 0.40f)
            val u1X = v1X / span1
            val u1Z = v1Z / span1

            val v2X = p2AMid[0] - p2BMid[0]
            val v2Z = p2AMid[2] - p2BMid[2]
            val span2 = sqrt(v2X * v2X + v2Z * v2Z).coerceIn(0.045f, 0.40f)
            val u2X = v2X / span2
            val u2Z = v2Z / span2

            val camPose = frame.camera.pose
            val camAngle = atan2(camPose.tx() - pCenterMid[0], camPose.tz() - pCenterMid[2])
            val camDirX = sin(camAngle)
            val camDirZ = cos(camAngle)

            val dot0 = u1X * camDirX + u1Z * camDirZ
            val dot1 = -dot0
            val dot2 = u2X * camDirX + u2Z * camDirZ
            val dot3 = -dot2

            val dots = floatArrayOf(dot0, dot1, dot2, dot3)
            var bestIdx = 0
            var maxDot = -Float.MAX_VALUE
            for (i in 0 until 4) {
                if (dots[i] > maxDot) {
                    maxDot = dots[i]
                    bestIdx = i
                }
            }

            val margin = 1.06f
            val targetW: Float
            val targetD: Float
            val targetHeading: Float

            when (bestIdx) {
                0 -> {
                    targetD = (span1 * margin).coerceIn(0.045f, 0.40f)
                    targetW = (span2 * margin).coerceIn(0.045f, 0.40f)
                    targetHeading = atan2(u1X, u1Z)
                }
                1 -> {
                    targetD = (span1 * margin).coerceIn(0.045f, 0.40f)
                    targetW = (span2 * margin).coerceIn(0.045f, 0.40f)
                    targetHeading = atan2(-u1X, -u1Z)
                }
                2 -> {
                    targetD = (span2 * margin).coerceIn(0.045f, 0.40f)
                    targetW = (span1 * margin).coerceIn(0.045f, 0.40f)
                    targetHeading = atan2(u2X, u2Z)
                }
                else -> {
                    targetD = (span2 * margin).coerceIn(0.045f, 0.40f)
                    targetW = (span1 * margin).coerceIn(0.045f, 0.40f)
                    targetHeading = atan2(-u2X, -u2Z)
                }
            }

            var targetH = (min(targetW, targetD) * 0.44f).coerceIn(0.032f, 0.14f)
            try {
                val pc = frame.acquirePointCloud()
                val pBuf = pc.points
                val pCount = pBuf.remaining() / 4
                var maxY = 0f
                var validPts = 0
                val radiusLimit = max(targetW, targetD) * 0.65f
                for (p in 0 until pCount) {
                    if (pBuf.get(p * 4 + 3) < 0.2f) continue
                    val px = pBuf.get(p * 4) - pCenterMid[0]
                    val py = pBuf.get(p * 4 + 1) - planeY
                    val pz = pBuf.get(p * 4 + 2) - pCenterMid[2]
                    if (py in 0.008f..0.28f && sqrt(px * px + pz * pz) <= radiusLimit) {
                        if (py > maxY) maxY = py
                        validPts++
                    }
                }
                pc.close()
                if (validPts >= 3 && maxY > 0.015f) {
                    targetH = (maxY * 1.15f).coerceIn(0.032f, 0.22f)
                }
            } catch (e: Exception) {}

            if (!hasLiveCling) {
                liveWorldX = pCenterMid[0]
                liveWorldY = planeY
                liveWorldZ = pCenterMid[2]
                liveWidth = targetW
                liveDepth = targetD
                liveHeight = targetH
                liveAzimuth = targetHeading
                liveCos2A = cos(targetHeading * 2f)
                liveSin2A = sin(targetHeading * 2f)
                liveClingScreenX = boxScreenCX
                liveClingScreenY = boxScreenCY
                hasLiveCling = true
            } else {
                val alphaPos = 0.35f
                val alphaDim = 0.25f
                liveWorldX += alphaPos * (pCenterMid[0] - liveWorldX)
                liveWorldY = planeY
                liveWorldZ += alphaPos * (pCenterMid[2] - liveWorldZ)
                liveWidth += alphaDim * (targetW - liveWidth)
                liveDepth += alphaDim * (targetD - liveDepth)
                liveHeight += alphaDim * (targetH - liveHeight)

                liveCos2A = liveCos2A * 0.75f + cos(targetHeading * 2f) * 0.25f
                liveSin2A = liveSin2A * 0.75f + sin(targetHeading * 2f) * 0.25f
                var smoothedAxis = 0.5f * atan2(liveSin2A, liveCos2A)
                if (sin(smoothedAxis) * camDirX + cos(smoothedAxis) * camDirZ < 0f) {
                    smoothedAxis += Math.PI.toFloat()
                }
                liveAzimuth = smoothedAxis

                liveClingScreenX += alphaPos * (boxScreenCX - liveClingScreenX)
                liveClingScreenY += alphaPos * (boxScreenCY - liveClingScreenY)
            }
            liveClingConfidence = (liveClingConfidence + 1).coerceAtMost(6)

        } catch (e: Exception) {
        } finally {
            cameraImage?.close()
        }
    }

    // =========================================================================
    // PHASE 2: PLANE-ATTACHED ANCHOR LOCK + CLEAN MAGICTOUCH REFINEMENT
    // =========================================================================
    private fun getActiveRootPlane(): Plane? {
        var p = trackedTablePlane
        while (p?.subsumedBy != null) {
            p = p.subsumedBy
        }
        if (p != null && p.trackingState == TrackingState.TRACKING) {
            trackedTablePlane = p
            return p
        }
        return null
    }

    /**
     * Creates a strictly gravity-aligned (Y-up [0,0,0,1]) anchor attached directly
     * to the tracked horizontal table Plane whenever available so vibration recovery
     * keeps the prism glued to the table's visual map.
     */
    private fun createPlaneGroundedAnchor(x: Float, y: Float, z: Float): Anchor? {
        val gravityUpPose = Pose(floatArrayOf(x, y, z), floatArrayOf(0f, 0f, 0f, 1f))
        val rootPlane = getActiveRootPlane()
        if (rootPlane != null) {
            try {
                val planePose = Pose(
                    floatArrayOf(x, rootPlane.centerPose.ty(), z),
                    floatArrayOf(0f, 0f, 0f, 1f)
                )
                return rootPlane.createAnchor(planePose)
            } catch (_: Exception) {}
        }
        return try {
            sceneView.session?.createAnchor(gravityUpPose)
        } catch (_: Exception) {
            null
        }
    }

    private fun placePrismAtHitTest() {
        val frame = lastFrame ?: return
        val hitResults = frame.hitTest(sceneView.width / 2f, sceneView.height / 2f)
        val planeHit = hitResults.firstOrNull {
            val t = it.trackable
            t is Plane && t.trackingState == TrackingState.TRACKING && t.type == Plane.Type.HORIZONTAL_UPWARD_FACING
        }
        val firstHit = planeHit ?: hitResults.firstOrNull()

        if (firstHit != null) {
            if (planeHit != null) {
                trackedTablePlane = planeHit.trackable as Plane
            }
            val hitPose = firstHit.hitPose
            val seedScreenX: Float
            val seedScreenY: Float
            val anchor: Anchor?

            if (hasLiveCling) {
                anchor = createPlaneGroundedAnchor(liveWorldX, liveWorldY, liveWorldZ)
                prismWidth = liveWidth
                prismDepth = liveDepth
                prismHeight = liveHeight
                initialAzimuth = liveAzimuth
                seedScreenX = liveClingScreenX
                seedScreenY = liveClingScreenY
            } else {
                anchor = createPlaneGroundedAnchor(hitPose.tx(), hitPose.ty(), hitPose.tz())
                val camPose = frame.camera.pose
                initialAzimuth = atan2(camPose.tx() - hitPose.tx(), camPose.tz() - hitPose.tz())
                seedScreenX = sceneView.width / 2f
                seedScreenY = sceneView.height / 2f
            }

            if (anchor == null) {
                Toast.makeText(this, "Tracking not ready — move slightly", Toast.LENGTH_SHORT).show()
                return
            }

            // Hide the pre-lock GPU ring immediately
            liveRingRoot?.isVisible = false
            isBoxPlaced = true

            // Snapshot VP matrix before spawning prism so MagicTouch gets a clean, overlay-free frame
            val vMat = FloatArray(16)
            val pMat = FloatArray(16)
            val vpMat = FloatArray(16)
            val invVpMat = FloatArray(16)

            frame.camera.getViewMatrix(vMat, 0)
            frame.camera.getProjectionMatrix(pMat, 0, 0.05f, 50.0f)
            Matrix.multiplyMM(vpMat, 0, pMat, 0, vMat, 0)
            Matrix.invertM(invVpMat, 0, vpMat, 0)

            val snapshotAnchorPose = anchor.pose
            val camPose = frame.camera.pose
            val camInAnchor = snapshotAnchorPose.inverse().transformPoint(
                floatArrayOf(camPose.tx(), camPose.ty(), camPose.tz())
            )
            val camAngle = atan2(camInAnchor[0], camInAnchor[2])

            attachPrismToAnchor(anchor)

            // Briefly hide the new GPU prism for 2 frames (45ms) so MagicTouch receives a clean camera image
            isTakingCleanSnapshot = true
            prismAnchorNode?.isVisible = false
            mainHandler.postDelayed({
                runMagicTouchPipeline(
                    invVpMat,
                    snapshotAnchorPose,
                    camAngle,
                    seedScreenX,
                    seedScreenY,
                    prismHeight
                )
            }, 45L)

            actionButton.visibility = View.GONE
            scaleBar.visibility = View.VISIBLE
            previewScrollView.visibility = View.VISIBLE
            forceSnapBtn.visibility = View.VISIBLE
            forceSnapBtn.text = "FORCE SNAP [${sequence[currentStepIdx].id.uppercase(Locale.US)}]"
            targetBadge.text = sequence[currentStepIdx].label
            updateChecklistUI()
            updateScaleLabels()
            overlayView.showCrosshair = true
            overlayView.postInvalidate()

            try {
                toneGen?.startTone(ToneGenerator.TONE_PROP_ACK, 140)
            } catch (e: Exception) {}
        } else {
            Toast.makeText(this, "Aim reticle at table surface first", Toast.LENGTH_SHORT).show()
        }
    }

    private fun runMagicTouchPipeline(
        invVpMatrix: FloatArray,
        snapshotAnchorPose: Pose,
        camAngle: Float,
        seedScreenX: Float,
        seedScreenY: Float,
        seedHeightM: Float
    ) {
        val sWidth = sceneView.width
        val sHeight = sceneView.height
        if (sWidth <= 0 || sHeight <= 0) {
            isTakingCleanSnapshot = false
            prismAnchorNode?.isVisible = true
            return
        }

        val bitmap = Bitmap.createBitmap(sWidth, sHeight, Bitmap.Config.ARGB_8888)
        PixelCopy.request(
            sceneView as SurfaceView,
            bitmap,
            PixelCopy.OnPixelCopyFinishedListener { copyResult ->
                // Restore GPU prism immediately once clean frame is grabbed
                isTakingCleanSnapshot = false
                prismAnchorNode?.isVisible = true

                if (copyResult == PixelCopy.SUCCESS) {
                    Thread {
                        val interp = tfliteInterpreter
                        if (interp == null) {
                            logDebug("⚠️ MagicTouch TFLite not ready — keeping GPU 3D Ring lock")
                            return@Thread
                        }

                        try {
                            val startMs = System.currentTimeMillis()
                            val inTensor = interp.getInputTensor(0)
                            val inShape = inTensor.shape()
                            val inType = inTensor.dataType()
                            val inH = if (inShape.size >= 3) inShape[1] else 512
                            val inW = if (inShape.size >= 3) inShape[2] else 512
                            val inC = if (inShape.size >= 4) inShape[3] else 4

                            val minDim = min(bitmap.width, bitmap.height)
                            val cropX = (bitmap.width - minDim) / 2f
                            val cropY = (bitmap.height - minDim) / 2f
                            val squareBmp = Bitmap.createBitmap(bitmap, cropX.toInt(), cropY.toInt(), minDim, minDim)
                            val scaledBmp = Bitmap.createScaledBitmap(squareBmp, inW, inH, true)

                            val normKeyX = ((seedScreenX - cropX) / minDim).coerceIn(0.15f, 0.85f)
                            val normKeyY = ((seedScreenY - cropY) / minDim).coerceIn(0.15f, 0.85f)
                            val keyPxX = normKeyX * inW
                            val keyPxY = normKeyY * inH
                            val invTwoSigmaSq = 1.0f / (2.0f * 12.0f * 12.0f)

                            val inputBuffer = ByteBuffer.allocateDirect(inTensor.numBytes()).apply {
                                order(ByteOrder.nativeOrder())
                            }

                            val intValues = IntArray(inW * inH)
                            scaledBmp.getPixels(intValues, 0, inW, 0, 0, inW, inH)

                            var pIdx = 0
                            for (y in 0 until inH) {
                                val dy = y - keyPxY
                                for (x in 0 until inW) {
                                    val px = intValues[pIdx++]
                                    val r = (px shr 16) and 0xFF
                                    val g = (px shr 8) and 0xFF
                                    val b = px and 0xFF

                                    val dx = x - keyPxX
                                    val gauss = exp(-(dx * dx + dy * dy) * invTwoSigmaSq)

                                    if (inType == DataType.FLOAT32) {
                                        inputBuffer.putFloat(r / 255.0f)
                                        inputBuffer.putFloat(g / 255.0f)
                                        inputBuffer.putFloat(b / 255.0f)
                                        if (inC >= 4) inputBuffer.putFloat(gauss)
                                    } else {
                                        inputBuffer.put(r.toByte())
                                        inputBuffer.put(g.toByte())
                                        inputBuffer.put(b.toByte())
                                        if (inC >= 4) inputBuffer.put((gauss * 255f).toInt().coerceIn(0, 255).toByte())
                                    }
                                }
                            }
                            inputBuffer.rewind()

                            val outTensor = interp.getOutputTensor(0)
                            val outShape = outTensor.shape()
                            val outType = outTensor.dataType()
                            val outputBuffer = ByteBuffer.allocateDirect(outTensor.numBytes()).apply {
                                order(ByteOrder.nativeOrder())
                            }

                            interp.run(inputBuffer, outputBuffer)
                            outputBuffer.rewind()

                            val latencyMs = System.currentTimeMillis() - startMs
                            logDebug("MagicTouch TFLite: ${latencyMs}ms | Out:${outShape.contentToString()}")

                            projectMagicTouchBufferAndOrient(
                                outputBuffer, outShape, outType,
                                cropX, cropY, minDim.toFloat(),
                                normKeyX, normKeyY,
                                invVpMatrix, snapshotAnchorPose,
                                camAngle, seedHeightM, sWidth, sHeight
                            )
                        } catch (e: Exception) {
                            logDebug("❌ MagicTouch Exception: ${e.javaClass.simpleName}: ${e.message}")
                        }
                    }.start()
                }
            },
            mainHandler
        )
    }

    private fun projectMagicTouchBufferAndOrient(
        outputBuffer: ByteBuffer,
        outShape: IntArray,
        outType: DataType,
        cropX: Float,
        cropY: Float,
        minDim: Float,
        normKeyX: Float,
        normKeyY: Float,
        invVpMatrix: FloatArray,
        snapshotAnchorPose: Pose,
        camAngle: Float,
        seedHeightM: Float,
        sWidth: Int,
        sHeight: Int
    ) {
        val mH: Int
        val mW: Int
        val mC: Int

        when (outShape.size) {
            4 -> {
                if (outShape[3] <= 4) {
                    mH = outShape[1]; mW = outShape[2]; mC = outShape[3]
                } else {
                    mC = outShape[1]; mH = outShape[2]; mW = outShape[3]
                }
            }
            3 -> {
                mH = outShape[1]; mW = outShape[2]; mC = 1
            }
            else -> return
        }

        val scores = FloatArray(mW * mH)
        if (outType == DataType.FLOAT32) {
            val fb = outputBuffer.asFloatBuffer()
            for (i in 0 until (mW * mH)) {
                scores[i] = if (mC == 1) {
                    fb.get(i)
                } else {
                    fb.get(i * mC + (mC - 1)) - fb.get(i * mC)
                }
            }
        } else {
            for (i in 0 until (mW * mH)) {
                val base = i * mC + (mC - 1)
                scores[i] = (outputBuffer.get(base).toInt() and 0xFF) / 255.0f
            }
        }

        val seedX = (normKeyX * mW).toInt().coerceIn(8, mW - 9)
        val seedY = (normKeyY * mH).toInt().coerceIn(8, mH - 9)
        var seedSum = 0f; var seedCnt = 0
        for (dy in -4..4) {
            for (dx in -4..4) {
                seedSum += scores[(seedY + dy) * mW + (seedX + dx)]
                seedCnt++
            }
        }
        val seedMean = seedSum / seedCnt

        var borderSum = 0f; var borderCnt = 0
        for (x in 0 until mW step 4) {
            borderSum += scores[2 * mW + x] + scores[(mH - 3) * mW + x]
            borderCnt += 2
        }
        val borderMean = borderSum / borderCnt.coerceAtLeast(1)

        if (abs(seedMean - borderMean) < 0.04f) {
            logDebug("ℹ️ MagicTouch low contrast — keeping GPU 3D Ring lock")
            return
        }

        val thresh = borderMean + (seedMean - borderMean) * 0.48f
        val seedIsHigher = seedMean > borderMean

        val gSize = 128
        val gridMask = BooleanArray(gSize * gSize)
        for (gy in 1 until gSize - 1) {
            val my = (gy * mH) / gSize
            val rowOff = my * mW
            for (gx in 1 until gSize - 1) {
                val mx = (gx * mW) / gSize
                val s = scores[rowOff + mx]
                gridMask[gy * gSize + gx] = if (seedIsHigher) (s >= thresh) else (s <= thresh)
            }
        }

        val seedGX = (normKeyX * gSize).toInt().coerceIn(4, gSize - 5)
        val seedGY = (normKeyY * gSize).toInt().coerceIn(4, gSize - 5)
        var startIdx = -1
        var bestDist = Int.MAX_VALUE
        for (gy in (seedGY - 10)..(seedGY + 10)) {
            for (gx in (seedGX - 10)..(seedGX + 10)) {
                val idx = gy * gSize + gx
                if (gridMask[idx]) {
                    val d = (gx - seedGX) * (gx - seedGX) + (gy - seedGY) * (gy - seedGY)
                    if (d < bestDist) {
                        bestDist = d
                        startIdx = idx
                    }
                }
            }
        }

        if (startIdx == -1) return

        val visited = BooleanArray(gSize * gSize)
        val queue = ArrayDeque<Int>()
        queue.add(startIdx)
        visited[startIdx] = true

        val fgScreenX = mutableListOf<Float>()
        val fgScreenY = mutableListOf<Float>()
        var sumX = 0f
        var sumY = 0f

        while (!queue.isEmpty()) {
            val curr = queue.removeFirst()
            val cx = curr % gSize
            val cy = curr / gSize

            val sx = cropX + ((cx + 0.5f) / gSize) * minDim
            val sy = cropY + ((cy + 0.5f) / gSize) * minDim
            fgScreenX.add(sx)
            fgScreenY.add(sy)
            sumX += sx
            sumY += sy

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
        if (count < 30 || count > (gSize * gSize * 0.72f).toInt()) {
            logDebug("ℹ️ Keeping GPU 3D Ring Fit (MagicTouch mask count: $count)")
            return
        }

        val barX = sumX / count
        val barY = sumY / count

        var bestArea = Float.MAX_VALUE
        var bestRad = 0f
        for (deg in 0 until 90) {
            val rad = Math.toRadians(deg.toDouble()).toFloat()
            val c = cos(rad)
            val s = sin(rad)
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
            if (area < bestArea) {
                bestArea = area
                bestRad = rad
            }
        }

        val uX = cos(bestRad)
        val uY = sin(bestRad)
        val vX = -uY
        val vY = uX

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
        val halfSpan1Px = ((maxU - minU) * 0.5f).coerceAtLeast(16f)
        val halfSpan2Px = ((maxV - minV) * 0.5f).coerceAtLeast(16f)

        val centerScreenX = barX + midU * uX + midV * vX
        val centerScreenY = barY + midU * uY + midV * vY

        val invSnapshotAnchor = snapshotAnchorPose.inverse()
        val planeY = snapshotAnchorPose.ty()
        val midPlaneY = planeY + seedHeightM * 0.5f

        val pCenterMid = intersectScreenRayWithHorizontalPlane(centerScreenX, centerScreenY, sWidth.toFloat(), sHeight.toFloat(), invVpMatrix, midPlaneY)
        val p1AMid = intersectScreenRayWithHorizontalPlane(centerScreenX + halfSpan1Px * uX, centerScreenY + halfSpan1Px * uY, sWidth.toFloat(), sHeight.toFloat(), invVpMatrix, midPlaneY)
        val p1BMid = intersectScreenRayWithHorizontalPlane(centerScreenX - halfSpan1Px * uX, centerScreenY - halfSpan1Px * uY, sWidth.toFloat(), sHeight.toFloat(), invVpMatrix, midPlaneY)
        val p2AMid = intersectScreenRayWithHorizontalPlane(centerScreenX + halfSpan2Px * vX, centerScreenY + halfSpan2Px * vY, sWidth.toFloat(), sHeight.toFloat(), invVpMatrix, midPlaneY)
        val p2BMid = intersectScreenRayWithHorizontalPlane(centerScreenX - halfSpan2Px * vX, centerScreenY - halfSpan2Px * vY, sWidth.toFloat(), sHeight.toFloat(), invVpMatrix, midPlaneY)

        if (pCenterMid == null || p1AMid == null || p1BMid == null || p2AMid == null || p2BMid == null) return

        val pCenterTable = floatArrayOf(pCenterMid[0], planeY, pCenterMid[2])
        val lCenter = invSnapshotAnchor.transformPoint(pCenterTable)
        val l1A = invSnapshotAnchor.transformPoint(floatArrayOf(p1AMid[0], planeY, p1AMid[2]))
        val l1B = invSnapshotAnchor.transformPoint(floatArrayOf(p1BMid[0], planeY, p1BMid[2]))
        val l2A = invSnapshotAnchor.transformPoint(floatArrayOf(p2AMid[0], planeY, p2AMid[2]))
        val l2B = invSnapshotAnchor.transformPoint(floatArrayOf(p2BMid[0], planeY, p2BMid[2]))

        val v1X = l1A[0] - l1B[0]
        val v1Z = l1A[2] - l1B[2]
        val span1 = sqrt(v1X * v1X + v1Z * v1Z).coerceIn(0.04f, 0.45f)
        val u1X = v1X / span1
        val u1Z = v1Z / span1

        val v2X = l2A[0] - l2B[0]
        val v2Z = l2A[2] - l2B[2]
        val span2 = sqrt(v2X * v2X + v2Z * v2Z).coerceIn(0.04f, 0.45f)
        val u2X = v2X / span2
        val u2Z = v2Z / span2

        val camDirX = sin(camAngle)
        val camDirZ = cos(camAngle)

        val dot0 = u1X * camDirX + u1Z * camDirZ
        val dot1 = -dot0
        val dot2 = u2X * camDirX + u2Z * camDirZ
        val dot3 = -dot2

        val dots = floatArrayOf(dot0, dot1, dot2, dot3)
        var bestIdx = 0
        var maxDot = -Float.MAX_VALUE
        for (i in 0 until 4) {
            if (dots[i] > maxDot) {
                maxDot = dots[i]
                bestIdx = i
            }
        }

        val margin = 1.06f
        val finalWidth: Float
        val finalDepth: Float
        val headingAngle: Float

        when (bestIdx) {
            0 -> {
                finalDepth = (span1 * margin + 0.006f).coerceIn(0.04f, 0.45f)
                finalWidth = (span2 * margin + 0.006f).coerceIn(0.04f, 0.45f)
                headingAngle = atan2(u1X, u1Z)
            }
            1 -> {
                finalDepth = (span1 * margin + 0.006f).coerceIn(0.04f, 0.45f)
                finalWidth = (span2 * margin + 0.006f).coerceIn(0.04f, 0.45f)
                headingAngle = atan2(-u1X, -u1Z)
            }
            2 -> {
                finalDepth = (span2 * margin + 0.006f).coerceIn(0.04f, 0.45f)
                finalWidth = (span1 * margin + 0.006f).coerceIn(0.04f, 0.45f)
                headingAngle = atan2(u2X, u2Z)
            }
            else -> {
                finalDepth = (span2 * margin + 0.006f).coerceIn(0.04f, 0.45f)
                finalWidth = (span1 * margin + 0.006f).coerceIn(0.04f, 0.45f)
                headingAngle = atan2(-u2X, -u2Z)
            }
        }

        runOnUiThread {
            initialAzimuth = headingAngle
            prismWidth = finalWidth * 0.65f + prismWidth * 0.35f
            prismDepth = finalDepth * 0.65f + prismDepth * 0.35f

            val currentAnchor = prismAnchor
            if (currentAnchor != null) {
                val curPose = currentAnchor.pose
                val deltaX = (lCenter[0] * 0.5f).coerceIn(-0.06f, 0.06f)
                val deltaZ = (lCenter[2] * 0.5f).coerceIn(-0.06f, 0.06f)
                val worldCenterNow = curPose.transformPoint(floatArrayOf(deltaX, 0f, deltaZ))
                val refinedAnchor = createPlaneGroundedAnchor(worldCenterNow[0], curPose.ty(), worldCenterNow[2])
                if (refinedAnchor != null) {
                    attachPrismToAnchor(refinedAnchor)
                } else {
                    syncPrismYawTransform()
                    rebuildPrismGpuMeshes()
                }
            }

            updateScaleLabels()

            val wCm = (prismWidth * 100).roundToInt()
            val dCm = (prismDepth * 100).roundToInt()
            val hCm = (prismHeight * 100).roundToInt()
            val yawDeg = Math.toDegrees(headingAngle.toDouble()).roundToInt()

            logDebug("✓ MagicTouch GPU Lock: ${wCm}x${dCm}x${hCm}cm | Heading: ${yawDeg}°")
            Toast.makeText(
                this@MainActivity,
                "✓ MagicTouch GPU Locked: ${wCm}×${dCm}×${hCm}cm",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    private fun intersectScreenRayWithHorizontalPlane(
        screenX: Float, screenY: Float,
        screenWidth: Float, screenHeight: Float,
        invVpMatrix: FloatArray,
        planeY: Float
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
        if (len < 1e-4f || abs(dirY / len) < 1e-4f) return null

        val dX = dirX / len
        val dY = dirY / len
        val dZ = dirZ / len

        val t = (planeY - rOy) / dY
        if (t <= 0.05f) return null

        return floatArrayOf(rOx + t * dX, planeY, rOz + t * dZ)
    }

    // =========================================================================
    // PER-FRAME GPU SYNC & THROTTLED HUD TRACKING LOOP
    // =========================================================================
    private fun onTrackingFrame(frame: Frame) {
        val camera = frame.camera
        if (camera.trackingState != TrackingState.TRACKING) return

        val now = System.currentTimeMillis()
        val shouldUpdateHud = (now - lastHudUpdateMs) >= 100L

        if (!isBoxPlaced) {
            val hitResults = frame.hitTest(sceneView.width / 2f, sceneView.height / 2f)
            val planeHit = hitResults.firstOrNull {
                val t = it.trackable
                t is Plane && t.trackingState == TrackingState.TRACKING && t.type == Plane.Type.HORIZONTAL_UPWARD_FACING
            }
            val hit = planeHit ?: hitResults.firstOrNull()

            if (hit != null) {
                if (planeHit != null) {
                    trackedTablePlane = planeHit.trackable as Plane
                }
                frameCounter++
                if (frameCounter % 2 == 0) {
                    updateLiveClingReticle(frame, hit.hitPose)
                }
            } else {
                hasLiveCling = false
                liveClingConfidence = 0
            }

            // Update GPU 3D Volumetric Ring transform directly on the render frame (0 ms lag!)
            if (gpuMaterialsReady) {
                val ring = liveRingRoot
                if (ring != null) {
                    if (hasLiveCling) {
                        val rx = (liveWidth * 0.68f).coerceIn(0.032f, 0.26f)
                        val rz = (liveDepth * 0.68f).coerceIn(0.032f, 0.26f)
                        val ringH = liveHeight.coerceIn(0.032f, 0.22f)
                        ring.isVisible = true
                        ring.position = Position(liveWorldX, liveWorldY, liveWorldZ)
                        ring.rotation = Rotation(0f, Math.toDegrees(liveAzimuth.toDouble()).toFloat(), 0f)
                        ring.scale = Scale(rx, ringH, rz)
                        if (!liveRingIsClingColor) {
                            setNodeMaterial(liveRingWallNode, matRingClingWall)
                            setNodeMaterial(liveRingWireNode, matRingClingWire)
                            liveRingIsClingColor = true
                        }
                    } else if (hit != null) {
                        val hp = hit.hitPose
                        ring.isVisible = true
                        ring.position = Position(hp.tx(), hp.ty(), hp.tz())
                        ring.rotation = Rotation(0f, 0f, 0f)
                        ring.scale = Scale(0.075f, 0.035f, 0.075f)
                        if (liveRingIsClingColor) {
                            setNodeMaterial(liveRingWallNode, matRingScanWall)
                            setNodeMaterial(liveRingWireNode, matRingScanWire)
                            liveRingIsClingColor = false
                        }
                    } else {
                        ring.isVisible = false
                    }
                }
            }

            if (shouldUpdateHud) {
                lastHudUpdateMs = now
                if (hasLiveCling) {
                    val wCm = (liveWidth * 100).roundToInt()
                    val dCm = (liveDepth * 100).roundToInt()
                    val hCm = (liveHeight * 100).roundToInt()
                    distIndicator.text = "◎ 3D RING ENCAPSULATED: ${wCm}×${dCm}×${hCm}cm — TAP LOCK"
                    distIndicator.setTextColor(Color.parseColor("#3FB950"))
                    valScale.text = "RING: ${wCm}×${dCm}"
                } else if (hit != null) {
                    distIndicator.text = "SURFACE LOCKED: AIM 3D RING AT ASSET"
                    distIndicator.setTextColor(Color.parseColor("#58A6FF"))
                } else {
                    distIndicator.text = "SCANNING FOR TABLE..."
                    distIndicator.setTextColor(Color.parseColor("#E3B341"))
                }
            }
            return
        }

        if (currentStepIdx >= sequence.size) return

        val anchor = prismAnchor ?: return
        if (anchor.trackingState != TrackingState.TRACKING) return

        val anchorPose = anchor.pose
        val camPose = camera.pose
        val step = sequence[currentStepIdx]

        val desc = getFacetDescriptor(currentStepIdx)
        val faceCenterWorld = localToWorld(desc.centerX, desc.centerY, desc.centerZ, anchorPose)
        val faceNormalWorld = normalToWorld(desc.normX, desc.normY, desc.normZ, anchorPose)

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

        // Hide reticle when looking from behind the active facet
        activeReticleNode?.isVisible = isFacingCamera && !isTakingCleanSnapshot

        // Throttled 10 Hz HUD updates (eliminates Android UI layout thrashing!)
        if (shouldUpdateHud) {
            lastHudUpdateMs = now
            valDist.text = "DIST: ${distCm.roundToInt()}cm"
            valAlign.text = "TILT: ${actualTiltDeg}° [${step.targetTilt}°]"

            if (distCm < 20f) {
                distIndicator.text = "⚠️ TOO CLOSE (${distCm.roundToInt()}cm) — STEP BACK"
                distIndicator.setTextColor(Color.parseColor("#F85149"))
            } else if (distCm > 38f) {
                distIndicator.text = "MOVE CLOSER (${distCm.roundToInt()}cm) — AIM 25-32cm"
                distIndicator.setTextColor(Color.parseColor("#E3B341"))
            } else {
                distIndicator.text = "✓ FOCUS DISTANCE: ${distCm.roundToInt()}cm (SHARP)"
                distIndicator.setTextColor(Color.parseColor("#3FB950"))
            }

            if (!isFacingCamera) {
                normalIndicator.text = "MOVE TO FRONT OF ${step.label}"
                normalIndicator.setTextColor(Color.parseColor("#F85149"))
                centerIndicator.text = "FACET FACING AWAY"
                centerIndicator.setTextColor(Color.parseColor("#8B949E"))
            } else {
                if (isNormalAligned) {
                    normalIndicator.text = "TILT: ${actualTiltDeg}° | ALIGN: ${String.format(Locale.US, "%.1f", normAngleErr)}° (LOCKED)"
                    normalIndicator.setTextColor(Color.parseColor("#3FB950"))
                } else {
                    normalIndicator.text = "TILT: ${actualTiltDeg}° [AIM ${step.targetTilt}°] | DEV: ${String.format(Locale.US, "%.1f", normAngleErr)}°"
                    normalIndicator.setTextColor(Color.parseColor("#F85149"))
                }

                if (isCentered) {
                    centerIndicator.text = "BULLSEYE CENTERED (OFFSET: ${String.format(Locale.US, "%.1f", centAngleErr)}°)"
                    centerIndicator.setTextColor(Color.parseColor("#3FB950"))
                } else {
                    centerIndicator.text = "AIM AT RETICLE (OFFSET: ${String.format(Locale.US, "%.1f", centAngleErr)}°)"
                    centerIndicator.setTextColor(Color.parseColor("#8B949E"))
                }
            }
        }

        if (isReadyToCapture && !isCapturing && now >= cooldownUntil) {
            updateGpuFacetMaterialsAndReticle(isAlignedNow = true)
            if (alignStartTime == null) {
                alignStartTime = now
            } else if (now - alignStartTime!! >= 450) {
                alignStartTime = null
                triggerLocalCapture()
            }
        } else {
            if (!isCapturing) {
                updateGpuFacetMaterialsAndReticle(isAlignedNow = false)
                alignStartTime = null
            }
        }
    }

    /**
     * Hides the GPU 3D cage for 45ms under cover of the shutter flash so every captured
     * JPEG plate saved for Modal 3D reconstruction is 100% clean and unobstructed.
     */
    private fun triggerLocalCapture() {
        if (isCapturing || currentStepIdx >= sequence.size) return
        isCapturing = true
        isTakingCleanSnapshot = true

        val snappedStepIdx = currentStepIdx
        val step = sequence[snappedStepIdx]

        playSnapFeedback()
        prismAnchorNode?.isVisible = false

        mainHandler.postDelayed({
            val sW = sceneView.width
            val sH = sceneView.height
            if (sW <= 0 || sH <= 0) {
                prismAnchorNode?.isVisible = true
                isTakingCleanSnapshot = false
                isCapturing = false
                return@postDelayed
            }

            val bitmap = Bitmap.createBitmap(sW, sH, Bitmap.Config.ARGB_8888)
            PixelCopy.request(
                sceneView as SurfaceView,
                bitmap,
                PixelCopy.OnPixelCopyFinishedListener { copyResult ->
                    // Immediately restore GPU prism visibility once PixelCopy completes
                    prismAnchorNode?.isVisible = true
                    isTakingCleanSnapshot = false

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
                                    updateGpuFacetMaterialsAndReticle(isAlignedNow = false, forceUpdate = true)

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
        }, 45L)
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

        snapFlash.alpha = 0.65f
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
        return anchorPose.transformPoint(rotated)
    }

    private fun normalToWorld(nx: Float, ny: Float, nz: Float, anchorPose: Pose): FloatArray {
        val rotated = rotateByAzimuth(nx, ny, nz, initialAzimuth)
        return anchorPose.rotateVector(rotated)
    }

    /**
     * Ultra-lightweight static 2D screen crosshair (no per-frame 3D projection or lag).
     */
    inner class CrosshairOverlayView(context: Context) : View(context) {
        var showCrosshair = false

        private val centerGuidePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(120, 255, 255, 255)
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val cx = width / 2f
            val cy = height / 2f
            if (showCrosshair) {
                canvas.drawCircle(cx, cy, 75f, centerGuidePaint)
                canvas.drawCircle(cx, cy, 8f, centerGuidePaint)
            } else {
                canvas.drawCircle(cx, cy, 6f, centerGuidePaint)
            }
        }
    }
}
