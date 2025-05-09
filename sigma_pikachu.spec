# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['sigma_pikachu/main.py'],
    pathex=[],
    binaries=[],
    datas=[('sigma_pikachu/pik64x64w.png', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='sigma_pikachu',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['sigma_pikachu.png'],
)
app = BUNDLE(
    exe,
    name='sigma_pikachu.app',
    icon='sigma_pikachu.png',
    bundle_identifier=None,
)
