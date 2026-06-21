// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ClaudeDJMascot",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "ClaudeDJMascot",
            path: "Sources/ClaudeDJMascot"
        )
    ]
)
