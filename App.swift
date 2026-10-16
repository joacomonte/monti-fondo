import SwiftUI
import Vision
import CoreImage
import CoreImage.CIFilterBuiltins
import UniformTypeIdentifiers

let ciContext = CIContext()

struct Item: Identifiable {
    let id = UUID()
    let url: URL
    let image: CIImage
    var mask: CIImage?    // máscara de Vision a resolución completa
    var error: String?
    var settings = Settings()  // ajustes de recorte propios de esta imagen
    var strokes: [Stroke] = []  // pincel manual
}

struct Stroke {
    var points: [CGPoint]   // normalizado 0–1, origen abajo-izquierda (como Core Image)
    let radius: CGFloat     // en píxeles de la imagen
    let include: Bool       // true = agregar, false = quitar
    var area: CIImage?      // garabato rasterizado (para mostrarlo mientras se procesa)
    var detected: CIImage?  // objeto/parte que Vision segmentó a partir del garabato
}

// Dibuja el trazo en blanco sobre negro, en coordenadas de un bitmap de w×h
func draw(_ stroke: Stroke, in ctx: CGContext, w: Int, h: Int, scale: CGFloat) {
    let r = stroke.radius * scale
    let pts = stroke.points.map { CGPoint(x: $0.x * CGFloat(w), y: $0.y * CGFloat(h)) }
    ctx.setFillColor(gray: 1, alpha: 1)
    ctx.setStrokeColor(gray: 1, alpha: 1)
    ctx.setLineCap(.round)
    ctx.setLineJoin(.round)
    ctx.setLineWidth(r * 2)
    ctx.fillEllipse(in: CGRect(x: pts[0].x - r, y: pts[0].y - r, width: r * 2, height: r * 2))
    ctx.addLines(between: pts)
    ctx.strokePath()
}

// Rasteriza un trazo a ≤2048 px y lo escala al tamaño de la imagen
func renderStroke(_ stroke: Stroke, size: CGSize) -> CIImage? {
    let scale = min(1, 2048 / max(size.width, size.height))
    let w = Int(size.width * scale), h = Int(size.height * scale)
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
    draw(stroke, in: ctx, w: w, h: h, scale: scale)
    guard let cg = ctx.makeImage() else { return nil }
    return CIImage(cgImage: cg).transformed(by: CGAffineTransform(scaleX: size.width / CGFloat(w), y: size.height / CGFloat(h)))
}

// Vision segmenta el objeto o la parte que señala el garabato (macOS 27)
func segment(_ image: CIImage, scribble stroke: Stroke) async throws -> CIImage? {
    let size = image.extent.size
    let scale = min(1, 2048 / max(size.width, size.height))
    let w = Int(size.width * scale), h = Int(size.height * scale)
    var buffer: CVPixelBuffer?
    CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_OneComponent8, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
    guard let buffer else { return nil }
    CVPixelBufferLockBaseAddress(buffer, [])
    if let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: w, height: h, bitsPerComponent: 8,
                           bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceGray(),
                           bitmapInfo: CGImageAlphaInfo.none.rawValue) {
        ctx.setFillColor(gray: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        draw(stroke, in: ctx, w: w, h: h, scale: scale)
    }
    CVPixelBufferUnlockBaseAddress(buffer, [])

    let request = GenerateIterativeSegmentationRequest(seedScribbleBuffer: CVReadOnlyPixelBuffer(unsafeBuffer: buffer))
    request.qualityLevel = .accurate
    if await request.assetStatus != .ready { try await request.downloadAssets() }
    guard let obs = try await request.perform(on: image) else { return nil }
    let mask = CIImage(cgImage: try obs.cgImage)
    return mask.transformed(by: CGAffineTransform(scaleX: size.width / mask.extent.width, y: size.height / mask.extent.height))
}

// La IA devuelve probabilidad 0–1; se pasa a sí/no (>0.3 = objeto) para que agregar/quitar sea completo.
// Al quitar, además se agranda 2 px para llevarse el halo del borde.
func harden(_ mask: CIImage, include: Bool) -> CIImage {
    let extent = mask.extent
    let gain: CGFloat = 20, bias = -0.3 * gain
    let v = CIVector(x: gain, y: 0, z: 0, w: 0)
    var m = mask.applyingFilter("CIColorMatrix", parameters: ["inputRVector": v, "inputGVector": v, "inputBVector": v,
                                                             "inputBiasVector": CIVector(x: bias, y: bias, z: bias, w: 0)])
        .applyingFilter("CIColorClamp")
    if !include {
        m = m.clampedToExtent().applyingFilter("CIMorphologyMaximum", parameters: [kCIInputRadiusKey: 2]).cropped(to: extent)
    }
    return m
}

// Máscara de Vision + objetos agregados − objetos quitados, en el orden de los trazos
func combinedMask(_ item: Item) -> CIImage? {
    guard var mask = item.mask else { return nil }
    for stroke in item.strokes {
        guard let m = stroke.detected else { continue }
        mask = stroke.include
            ? m.applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: mask])
            : m.applyingFilter("CIColorInvert").applyingFilter("CIMultiplyCompositing", parameters: [kCIInputBackgroundImageKey: mask])
    }
    return mask.cropped(to: item.image.extent)
}

func foregroundMask(_ url: URL) throws -> CIImage? {
    let handler = VNImageRequestHandler(url: url)
    let request = VNGenerateForegroundInstanceMaskRequest()
    try handler.perform([request])
    guard let result = request.results?.first else { return nil }
    let buffer = try result.generateScaledMaskForImage(forInstances: result.allInstances, from: handler)
    return CIImage(cvPixelBuffer: buffer)
}

struct Settings: Equatable {
    var threshold = 0.0  // 0–0.95: corta lo que Vision marcó con poca confianza
    var edge = 0.0       // px: negativo achica, positivo agranda
    var feather = 0.0    // px de difuminado del borde
}

func refine(_ mask: CIImage, _ s: Settings) -> CIImage {
    let extent = mask.extent
    var m = mask
    if s.threshold > 0 {
        // (m - t) / (1 - t), conserva la suavidad por encima del umbral
        let gain = CGFloat(1 / (1 - s.threshold)), bias = CGFloat(-s.threshold) * gain
        let f = CIFilter.colorMatrix()
        f.inputImage = m
        f.rVector = CIVector(x: gain, y: 0, z: 0, w: 0)
        f.gVector = CIVector(x: gain, y: 0, z: 0, w: 0)
        f.bVector = CIVector(x: gain, y: 0, z: 0, w: 0)
        f.biasVector = CIVector(x: bias, y: bias, z: bias, w: 0)
        m = f.outputImage!.applyingFilter("CIColorClamp")
    }
    if s.edge != 0 {
        let f = s.edge < 0 ? "CIMorphologyMinimum" : "CIMorphologyMaximum"
        m = m.clampedToExtent().applyingFilter(f, parameters: [kCIInputRadiusKey: abs(s.edge)]).cropped(to: extent)
    }
    if s.feather > 0 {
        m = m.clampedToExtent().applyingGaussianBlur(sigma: s.feather).cropped(to: extent)
    }
    return m
}

// Compresión 0–1, progresiva: colores 256→16 y resolución 100%→30% a la vez. 0 = PNG original
func compressionParams(_ c: Double) -> (colors: Int?, scale: Double) {
    if c <= 0 { return (nil, 1) }
    let colors = max(2, Int((256 * pow(16.0 / 256, c)).rounded()))
    return (colors, 1 - 0.7 * c)
}

// Puntos de muestra para estimar el peso mientras se mueve el slider
let sampleLevels = [0, 0.25, 0.5, 0.75, 1.0]

let pngquantPath = "/opt/homebrew/bin/pngquant"

func encode(_ image: CIImage, compression: Double) -> Data? {
    let (colors, scale) = compressionParams(compression)
    let out = scale < 1 ? image.applyingFilter("CILanczosScaleTransform", parameters: [kCIInputScaleKey: scale]) : image
    guard let png = ciContext.pngRepresentation(of: out, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!) else { return nil }
    guard let colors, FileManager.default.isExecutableFile(atPath: pngquantPath) else { return png }

    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let input = tmp.appendingPathExtension("png"), output = tmp.appendingPathExtension("q.png")
    defer { try? FileManager.default.removeItem(at: input); try? FileManager.default.removeItem(at: output) }
    try? png.write(to: input)
    let pq = Process()
    pq.executableURL = URL(fileURLWithPath: pngquantPath)
    pq.arguments = ["\(colors)", "--force", "--strip", "--output", output.path, input.path]
    try? pq.run()
    pq.waitUntilExit()
    return (try? Data(contentsOf: output)) ?? png
}

func cutout(_ item: Item, _ s: Settings) -> CIImage? {
    guard let mask = combinedMask(item) else { return nil }
    let f = CIFilter.blendWithMask()
    f.inputImage = item.image
    f.backgroundImage = CIImage.empty()
    f.maskImage = refine(mask, s)
    return f.outputImage?.cropped(to: item.image.extent)
}

@MainActor
final class Model: ObservableObject {
    @Published var items: [Item] = []
    @Published var selected: UUID?
    // Tamaño del panel (1 = 560 px de ancho), se recuerda entre aperturas
    @Published var panelScale = UserDefaults.standard.object(forKey: "panelScale") as? Double ?? 1 {
        didSet { UserDefaults.standard.set(panelScale, forKey: "panelScale") }
    }

    // Ajustes de la imagen seleccionada
    var settings: Settings {
        get { items.first { $0.id == selected }?.settings ?? Settings() }
        set { if let i = items.firstIndex(where: { $0.id == selected }) { items[i].settings = newValue } }
    }

    @Published var editing = false      // el pincel solo funciona en modo editar
    @Published var brushInclude = true   // garabato para agregar o quitar
    @Published var brushSize = 30.0      // diámetro en puntos de pantalla
    @Published var originalOpacity = 0.0 // imagen original debajo del recorte, para ver dónde pintar
    @Published var cursor: CGPoint?      // posición del pincel en la vista
    private var drawing = false

    // point: normalizado; pxPerPoint: píxeles de imagen por punto de pantalla
    func brush(at point: CGPoint, pxPerPoint: CGFloat) {
        guard editing, let i = items.firstIndex(where: { $0.id == selected }), items[i].mask != nil else { return }
        if drawing {
            items[i].strokes[items[i].strokes.count - 1].points.append(point)
        } else {
            drawing = true
            // ⌥ invierte el modo mientras se mantiene
            let include = NSEvent.modifierFlags.contains(.option) ? !brushInclude : brushInclude
            items[i].strokes.append(Stroke(points: [point], radius: brushSize / 2 * pxPerPoint, include: include))
        }
        let n = items[i].strokes.count - 1
        items[i].strokes[n].area = renderStroke(items[i].strokes[n], size: items[i].image.extent.size)
    }

    func endStroke() {
        guard drawing else { return }
        drawing = false
        guard let i = items.firstIndex(where: { $0.id == selected }), let stroke = items[i].strokes.last else { return }
        let id = items[i].id, n = items[i].strokes.count - 1, image = items[i].image
        status = "Analizando…"
        Task {
            let detected = try? await segment(image, scribble: stroke)
            guard let i = items.firstIndex(where: { $0.id == id }), n < items[i].strokes.count else { return }
            items[i].strokes[n].detected = detected.map { harden($0, include: stroke.include) } ?? CIImage(color: .black).cropped(to: image.extent)
            status = detected == nil ? "No se detectó nada ahí" : ""
        }
    }

    func undoStroke() {
        guard let i = items.firstIndex(where: { $0.id == selected }), !items[i].strokes.isEmpty else { return }
        items[i].strokes.removeLast()
    }

    // Flechas ←/→: pasar de imagen
    func step(_ delta: Int) {
        guard !items.isEmpty else { return }
        let i = items.firstIndex { $0.id == selected } ?? 0
        selected = items[(i + delta + items.count) % items.count].id
    }
    @Published var status = ""

    func add(_ urls: [URL]) {
        for url in urls {
            guard let image = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]) else { continue }
            let item = Item(url: url, image: image)
            items.append(item)
            selected = selected ?? item.id
            let id = item.id
            Task.detached {
                let start = Date()
                let result = Result { try foregroundMask(url) }
                let mask = try? result.get()
                let error: String? = switch result {
                case .failure(let e): e.localizedDescription
                case .success(nil): "Sin sujeto"
                case .success: nil
                }
                let ms = Int(Date().timeIntervalSince(start) * 1000)
                await MainActor.run {
                    guard let i = self.items.firstIndex(where: { $0.id == id }) else { return }
                    self.items[i].mask = mask ?? nil
                    self.items[i].error = error
                    self.status = "\(url.lastPathComponent): \(error ?? "\(ms) ms")"
                }
            }
        }
    }

    enum ExportStep { case none, destination, packaging, compression }
    enum Packaging { case loose, folder, zip }
    @Published var exportStep = ExportStep.none
    @Published var compression: [UUID: Double] = [:]  // por imagen, 0 = PNG original
    @Published var sizes: [UUID: Int] = [:]      // bytes reales con la compresión actual
    @Published var samples: [UUID: [Int]] = [:]  // bytes en cada sampleLevel, para estimar en vivo
    private var sizeLevel: [UUID: Double] = [:]  // compresión con la que se calculó sizes
    @Published var expanded: UUID?  // fila con el slider abierto por click
    @Published var hovered: UUID?
    private var exportDir: URL?  // nil = junto a cada original
    private var packaging = Packaging.loose
    private var sizesGeneration: [UUID: Int] = [:]

    func choosePackaging(_ p: Packaging) {
        packaging = p
        exportStep = .compression
        computeSamples()
        computeSizes()
    }

    // Codifica en memoria para mostrar el peso real que va a tener cada archivo
    func computeSizes(_ ids: [UUID]? = nil) {
        for item in items where ids?.contains(item.id) ?? true {
            let generation = (sizesGeneration[item.id] ?? 0) + 1
            sizesGeneration[item.id] = generation
            let c = compression[item.id] ?? 0
            Task.detached {
                guard let out = cutout(item, item.settings), let data = encode(out, compression: c) else { return }
                await MainActor.run {
                    guard generation == self.sizesGeneration[item.id] else { return }
                    self.sizes[item.id] = data.count
                    self.sizeLevel[item.id] = c
                }
            }
        }
    }

    // Codifica cada imagen en los niveles de muestra (una vez al entrar al paso)
    func computeSamples() {
        samples = [:]
        for item in items {
            Task.detached {
                guard let out = cutout(item, item.settings) else { return }
                let bytes = sampleLevels.map { encode(out, compression: $0)?.count ?? 0 }
                await MainActor.run { self.samples[item.id] = bytes }
            }
        }
    }

    // Peso exacto si ya se calculó para esta compresión; si no, interpolado entre muestras
    func displaySize(_ id: UUID) -> Int? {
        let c = compression[id] ?? 0
        if let exact = sizes[id], sizeLevel[id] == c { return exact }
        guard let s = samples[id] else { return nil }
        let i = min(Int(c / 0.25), sampleLevels.count - 2)
        let t = (c - sampleLevels[i]) / 0.25
        return Int(Double(s[i]) + (Double(s[i + 1]) - Double(s[i])) * t)
    }

    func chooseDestination(_ dir: URL?) {
        exportDir = dir
        exportStep = .packaging
    }

    func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Usar esta carpeta"
        NSApp.activate(ignoringOtherApps: true)
        if panel.runModal() == .OK { chooseDestination(panel.url) }
    }

    func export() {
        let name = "sin-fondo-" + Date().formatted(.iso8601.year().month().day().time(includingFractionalSeconds: false)).replacingOccurrences(of: ":", with: "")
        // Carpeta nueva / zip "junto al original" van en la carpeta de la primera imagen
        let base = exportDir ?? items[0].url.deletingLastPathComponent()
        let fm = FileManager.default
        let target: URL? = switch packaging {
        case .loose: exportDir
        case .folder: base.appendingPathComponent(name)
        case .zip: fm.temporaryDirectory.appendingPathComponent(name)
        }
        if let target, packaging != .loose { try? fm.createDirectory(at: target, withIntermediateDirectories: true) }

        var count = 0
        for item in items {
            guard let out = cutout(item, item.settings) else { continue }
            let dir = target ?? item.url.deletingLastPathComponent()
            let dest = dir.appendingPathComponent(item.url.deletingPathExtension().lastPathComponent + "-nobg.png")
            guard let data = encode(out, compression: compression[item.id] ?? 0) else { continue }
            try? data.write(to: dest)
            count += 1
        }

        var result = target ?? base
        if packaging == .zip, let target {
            result = base.appendingPathComponent(name + ".zip")
            let ditto = Process()
            ditto.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
            ditto.arguments = ["-c", "-k", "--keepParent", target.path, result.path]
            try? ditto.run()
            ditto.waitUntilExit()
            try? fm.removeItem(at: target)
        }
        status = "Exportadas \(count) imágenes"
        exportStep = .none
        NSWorkspace.shared.activateFileViewerSelecting([result])
    }
}

struct Checkerboard: View {
    var body: some View {
        Canvas { ctx, size in
            let s: CGFloat = 12
            for x in stride(from: 0, to: size.width, by: s) {
                for y in stride(from: 0, to: size.height, by: s) where (Int(x / s) + Int(y / s)) % 2 == 0 {
                    ctx.fill(Path(CGRect(x: x, y: y, width: s, height: s)), with: .color(.gray.opacity(0.3)))
                }
            }
        }
    }
}

struct ContentView: View {
    @ObservedObject var model: Model

    var current: Item? { model.items.first { $0.id == model.selected } }

    var preview: CGImage? {
        guard let item = current, var img = cutout(item, model.settings) else { return nil }
        // Garabatos que todavía se están analizando
        for stroke in item.strokes where stroke.detected == nil {
            guard let area = stroke.area else { continue }
            let color = stroke.include ? CIColor(red: 0.2, green: 0.9, blue: 0.4, alpha: 0.6) : CIColor(red: 1, green: 0.25, blue: 0.25, alpha: 0.6)
            let tinted = CIImage(color: color).cropped(to: item.image.extent).composited(over: img)
            img = tinted.applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: img, kCIInputMaskImageKey: area])
        }
        return ciContext.createCGImage(img, from: item.image.extent)
    }

    var original: CGImage? {
        guard let item = current, model.originalOpacity > 0 else { return nil }
        return ciContext.createCGImage(item.image, from: item.image.extent)
    }

    var body: some View {
        switch model.exportStep {
        case .none: editor
        case .destination:
            StepView(title: "¿Dónde guardar?", step: 1, backTitle: "Cancelar", back: { model.exportStep = .none }) {
                BigButton(icon: "arrow.down.circle", title: "Descargas", subtitle: "~/Downloads", id: "dest-default-button") { model.chooseDestination(.downloadsDirectory) }
                BigButton(icon: "folder", title: "Junto al original", subtitle: "En la carpeta de cada imagen", id: "dest-origin-button") { model.chooseDestination(nil) }
                BigButton(icon: "folder.badge.questionmark", title: "Elegir carpeta…", subtitle: "Abre el selector", id: "dest-pick-button") { model.pickFolder() }
            }
        case .packaging:
            StepView(title: "¿Cómo exportar?", step: 2, backTitle: "Atrás", back: { model.exportStep = .destination }) {
                BigButton(icon: "doc.on.doc", title: "Archivos sueltos", subtitle: "Un archivo por imagen", id: "pack-loose-button") { model.choosePackaging(.loose) }
                BigButton(icon: "folder.badge.plus", title: "Carpeta nueva", subtitle: "Todo en una carpeta", id: "pack-folder-button") { model.choosePackaging(.folder) }
                BigButton(icon: "doc.zipper", title: "ZIP", subtitle: "Todo en un .zip", id: "pack-zip-button") { model.choosePackaging(.zip) }
            }
        case .compression:
            StepView(title: "Tamaño", step: 3, backTitle: "Atrás", back: { model.exportStep = .packaging }) {
                // Orden fijo por peso original, para que las filas no salten al comprimir
                List(model.items.sorted { (model.samples[$0.id]?[0] ?? 0) > (model.samples[$1.id]?[0] ?? 0) }) { item in
                    SizeRow(model: model, item: item)
                }
                .accessibilityIdentifier("size-list")
                HStack {
                    Text("Total").bold()
                    Spacer()
                    Text(mb(model.items.compactMap { model.displaySize($0.id) }.reduce(0, +))).bold().monospacedDigit()
                }
                Text("Click o hover en una imagen para comprimirla")
                    .font(.callout).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
                Button { model.export() } label: {
                    Text("Exportar").font(.title3.weight(.semibold)).frame(maxWidth: .infinity).padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .accessibilityIdentifier("final-export-button")
            }
        }
    }

    var editor: some View {
        VStack(alignment: .leading, spacing: 10) {
            Slider(value: $model.panelScale, in: 0.7...1.6) {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
            }
            .controlSize(.mini)
            .accessibilityIdentifier("panel-size-slider")

            ZStack {
                Checkerboard()
                if let cg = preview, let item = current {
                    GeometryReader { geo in
                        let rect = fittedRect(item.image.extent.size, in: geo.size)
                        let pxPerPoint = item.image.extent.width / rect.width
                        ZStack {
                            if let orig = original {
                                Image(decorative: orig, scale: 1).resizable().scaledToFit().opacity(model.originalOpacity)
                            }
                            Image(decorative: cg, scale: 1).resizable().scaledToFit()
                        }
                        .frame(width: geo.size.width, height: geo.size.height)
                        .contentShape(Rectangle())
                        .gesture(DragGesture(minimumDistance: 0)
                            .onChanged { v in
                                guard model.editing else { return }
                                model.cursor = v.location
                                model.brush(at: CGPoint(x: (v.location.x - rect.minX) / rect.width,
                                                        y: 1 - (v.location.y - rect.minY) / rect.height), pxPerPoint: pxPerPoint)
                            }
                            .onEnded { _ in model.endStroke() })
                        .onContinuousHover { phase in
                            if case .active(let p) = phase, model.editing { model.cursor = p } else { model.cursor = nil }
                        }
                        if let c = model.cursor {
                            Circle()
                                .stroke(model.brushInclude ? Color.green : Color.red, lineWidth: 1.5)
                                .frame(width: model.brushSize, height: model.brushSize)
                                .position(c)
                                .allowsHitTesting(false)
                        }
                    }
                    .accessibilityIdentifier("preview-canvas")
                } else {
                    Text(current == nil ? "Arrastrá imágenes al ícono o acá" : "Procesando…").foregroundStyle(.secondary)
                }
            }
            .frame(height: 440 * model.panelScale)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .dropDestination(for: URL.self) { urls, _ in model.add(urls); return true }

            Toggle(isOn: $model.editing) { Label("Editar con pincel", systemImage: "paintbrush.pointed") }
                .toggleStyle(.button)
                .onChange(of: model.editing) { if !model.editing { model.cursor = nil } }
                .accessibilityIdentifier("edit-toggle")

            if model.editing {
                HStack {
                    Picker("Pincel", selection: $model.brushInclude) {
                        Label("Agregar", systemImage: "plus.circle").tag(true)
                        Label("Quitar", systemImage: "minus.circle").tag(false)
                    }
                    .pickerStyle(.segmented)
                    .fixedSize()
                    .accessibilityIdentifier("brush-mode-picker")
                    Text("⌥ invierte").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Deshacer") { model.undoStroke() }
                        .keyboardShortcut("z")
                        .disabled(current?.strokes.isEmpty ?? true)
                        .accessibilityIdentifier("undo-stroke-button")
                }
                Slider(value: $model.brushSize, in: 4...150) { Text("Pincel \(Int(model.brushSize)) pt").frame(width: 110, alignment: .leading) }
                    .accessibilityIdentifier("brush-size-slider")
                Slider(value: $model.originalOpacity, in: 0...1) { Text("Original \(Int(model.originalOpacity * 100))%").frame(width: 110, alignment: .leading) }
                    .accessibilityIdentifier("original-opacity-slider")
            }

            if !model.items.isEmpty {
                HStack {
                    Picker("Imagen", selection: $model.selected) {
                        ForEach(model.items) { item in
                            Text((item.error != nil ? "⚠️ " : "") + item.url.lastPathComponent).tag(Optional(item.id))
                        }
                    }
                    .labelsHidden()
                    .accessibilityIdentifier("image-picker")
                    Text("\(model.items.count)").foregroundStyle(.secondary)
                    Button("Limpiar") { model.items = []; model.selected = nil }
                        .accessibilityIdentifier("clear-button")
                }
            }

            DisclosureGroup("Opciones avanzadas") {
                Slider(value: $model.settings.threshold, in: 0...0.95) { Text("Umbral \(model.settings.threshold, specifier: "%.2f")").frame(width: 110, alignment: .leading) }
                    .accessibilityIdentifier("threshold-slider")
                Slider(value: $model.settings.edge, in: -10...10, step: 1) { Text("Borde \(Int(model.settings.edge)) px").frame(width: 110, alignment: .leading) }
                    .accessibilityIdentifier("edge-slider")
                Slider(value: $model.settings.feather, in: 0...10) { Text("Difuminado \(model.settings.feather, specifier: "%.1f")").frame(width: 110, alignment: .leading) }
                    .accessibilityIdentifier("feather-slider")
                HStack {
                    Spacer()
                    Button("Resetear") { model.settings = Settings() }.accessibilityIdentifier("reset-button")
                }
            }
            .accessibilityIdentifier("advanced-disclosure")

            Divider()

            HStack {
                Text(model.status).foregroundStyle(.secondary).lineLimit(1)
                Spacer()
                Button("Salir") { NSApp.terminate(nil) }.accessibilityIdentifier("quit-button")
                Button("Exportar…") { model.exportStep = .destination }
                    .keyboardShortcut(.defaultAction)
                    .disabled(model.items.isEmpty)
                    .accessibilityIdentifier("export-button")
            }
        }
        .padding()
        .frame(width: 560 * model.panelScale)
    }
}

@MainActor
struct SizeRow: View {
    @ObservedObject var model: Model
    let item: Item

    var body: some View {
        let c = model.compression[item.id] ?? 0
        let (colors, scale) = compressionParams(c)
        let active = model.expanded == item.id || model.hovered == item.id
        HStack(spacing: 10) {
            Text(item.url.deletingPathExtension().lastPathComponent).lineLimit(1).truncationMode(.middle)
            Spacer(minLength: 8)
            // Siempre ocupa su lugar (solo cambia la opacidad) para que la fila no cambie de tamaño
            Slider(value: Binding(get: { c }, set: { model.compression[item.id] = $0 }), in: 0...1,
                   onEditingChanged: { editing in if !editing { model.computeSizes([item.id]) } })
                .controlSize(.small)
                .frame(width: 130)
                .opacity(active ? 1 : 0)
                .disabled(!active)
                .help(colors.map { "\($0) colores · resolución \(Int(scale * 100))%" } ?? "PNG original")
                .accessibilityIdentifier("compression-slider-\(item.id)")
            Text(c > 0 ? "−\(Int(c * 100))%" : "")
                .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                .frame(width: 34, alignment: .trailing)
            Group {
                if let bytes = model.displaySize(item.id) { Text(mb(bytes)).monospacedDigit() }
                else { ProgressView().controlSize(.small) }
            }
            .frame(width: 72, alignment: .trailing)
        }
        .frame(height: 24)
        .contentShape(Rectangle())
        .onHover { model.hovered = $0 ? item.id : (model.hovered == item.id ? nil : model.hovered) }
        .onTapGesture { model.expanded = model.expanded == item.id ? nil : item.id }
        .accessibilityIdentifier("size-row-\(item.id)")
    }
}

// Rect que ocupa una imagen .scaledToFit dentro de un contenedor
func fittedRect(_ image: CGSize, in container: CGSize) -> CGRect {
    let scale = min(container.width / image.width, container.height / image.height)
    let size = CGSize(width: image.width * scale, height: image.height * scale)
    return CGRect(x: (container.width - size.width) / 2, y: (container.height - size.height) / 2, width: size.width, height: size.height)
}

func mb(_ bytes: Int) -> String {
    String(format: "%.2f MB", Double(bytes) / 1_000_000)
}

struct StepView<Content: View>: View {
    @AppStorage("panelScale") private var panelScale = 1.0
    let title: String
    let step: Int
    let backTitle: String
    let back: () -> Void
    @ViewBuilder let content: Content

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Button(backTitle, action: back)
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("export-back-button")
                Spacer()
                Text("Paso \(step) de 3").foregroundStyle(.secondary)
            }
            Text(title).font(.title2.bold()).frame(maxWidth: .infinity, alignment: .leading)
            content
        }
        .padding()
        .frame(width: 560 * panelScale, height: 560 * panelScale)
    }
}

struct BigButton: View {
    let icon: String
    let title: String
    let subtitle: String
    let id: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: icon).font(.title).frame(width: 36)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.title3.weight(.semibold))
                    Text(subtitle).font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(id)
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
    private let popover = NSPopover()
    private var model: Model!

    func applicationDidFinishLaunching(_ notification: Notification) {
        model = Model()
        popover.behavior = .applicationDefined  // no se cierra al hacer click afuera; se cierra con el ícono
        let hosting = NSHostingController(rootView: ContentView(model: model))
        hosting.sizingOptions = [.preferredContentSize]  // el popover toma el alto real del contenido
        popover.contentViewController = hosting

        let button = statusItem.button!
        button.image = NSImage(systemSymbolName: "person.crop.rectangle.badge.plus", accessibilityDescription: "monti-fondo")
        button.action = #selector(toggle)
        button.target = self
        // La ventana del status item reenvía el drag a su delegate
        button.window?.registerForDraggedTypes([.fileURL])
        button.window?.delegate = self

        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self, self.popover.isShown, self.model.exportStep == .none else { return event }
            switch event.keyCode {
            case 123: self.model.step(-1); return nil  // ←
            case 124: self.model.step(1); return nil   // →
            default: return event
            }
        }
    }

    @objc private func toggle() {
        if popover.isShown { popover.performClose(nil) } else { show() }
    }

    private func show() {
        guard let button = statusItem.button else { return }
        NSApp.activate(ignoringOtherApps: true)
        popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
    }

    func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        statusItem.button?.highlight(true)
        return .copy
    }

    func draggingExited(_ sender: NSDraggingInfo?) {
        statusItem.button?.highlight(false)
    }

    func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        statusItem.button?.highlight(false)
        let urls = sender.draggingPasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
        model.add(urls)
        show()
        return !urls.isEmpty
    }
}

@main
@MainActor
enum MontiFondoApp {
    static let delegate = AppDelegate()
    static func main() {
        let app = NSApplication.shared
        app.delegate = delegate
        app.run()
    }
}
