import AppKit
let size: CGFloat = 1024
let img = NSImage(size: NSSize(width: size, height: size))
img.lockFocus()
let rect = NSRect(x: 100, y: 100, width: 824, height: 824)
let path = NSBezierPath(roundedRect: rect, xRadius: 185, yRadius: 185)
NSGradient(starting: .systemTeal, ending: .systemIndigo)!.draw(in: path, angle: -90)
let cfg = NSImage.SymbolConfiguration(pointSize: 480, weight: .semibold)
    .applying(.init(paletteColors: [.white]))
let name = NSImage(systemSymbolName: "person.and.background.dotted", accessibilityDescription: nil) != nil ? "person.and.background.dotted" : "scissors"
let sym = NSImage(systemSymbolName: name, accessibilityDescription: nil)!.withSymbolConfiguration(cfg)!
let s = sym.size
sym.draw(in: NSRect(x: (size - s.width) / 2, y: (size - s.height) / 2, width: s.width, height: s.height))
img.unlockFocus()
let rep = NSBitmapImageRep(data: img.tiffRepresentation!)!
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
