fn main() {
    // This crate's bindings call hv_* directly; declare the framework here so its own
    // test binaries link, not only downstream consumers (vmm re-declares harmlessly).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Hypervisor");
    }
}
