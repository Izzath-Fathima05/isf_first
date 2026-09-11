"""
CAD-native STEP loader using OpenCascade (OCP).
loads a STEP file as an exact CAD B-Rep rather than converting it immediately to a triangle mesh.
shape = load_step("data/input/bowl.step")
"""

from pathlib import Path

try:
    from OCP.BRepCheck import BRepCheck_Analyzer  # type: ignore[import-not-found]
    from OCP.IFSelect import IFSelect_RetDone  # type: ignore[import-not-found]
    from OCP.STEPControl import STEPControl_Reader  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - runtime guard for missing OCP bindings
    raise RuntimeError(
        "OpenCascade Python bindings (OCP) are required to load STEP files. "
        "Install the package in the active Python environment."
    ) from exc


def load_step(filepath):
    """
    Load a STEP file as an OpenCascade TopoDS_Shape.
    Returns TopoDS_Shape, Exact OpenCascade B-Rep representation of the STEP geometry.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"STEP file not found: {filepath}"
        )

    if not filepath.is_file():
        raise ValueError(
            f"Input path is not a file: {filepath}"
        )
    reader = STEPControl_Reader()

    status = reader.ReadFile(str(filepath))

    if status != IFSelect_RetDone:
        raise ValueError(
            f"Failed to read STEP file:\n"
            f"    {filepath}\n"
            f"OpenCascade status code: {status}"
        )
    transfer_status = reader.TransferRoots()

    if not transfer_status:
        raise ValueError(
            f"OpenCascade could not transfer STEP geometry:\n"
            f"    {filepath}"
        )

    shape = reader.OneShape()

    if shape.IsNull():
        raise ValueError(
            f"STEP file was read, but the resulting CAD shape is null:\n"
            f"    {filepath}"
        )

    return shape


def validate_shape(shape):

    if shape is None:
        return False

    if shape.IsNull():
        return False

    analyzer = BRepCheck_Analyzer(shape)

    return analyzer.IsValid()


def shape_summary(shape):
    return {
        "is_null": shape.IsNull(),
        "is_valid": validate_shape(shape),
        "shape_type": shape.ShapeType(),
    }


def load_and_validate_step(filepath):
    shape = load_step(filepath)

    if not validate_shape(shape):
        raise ValueError(
            f"OpenCascade loaded the STEP file, but the resulting "
            f"B-Rep failed the geometry validity check:\n"
            f"    {filepath}"
        )

    return shape


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "    python src/cad_native/cad_loader.py "
            "<path_to_step>"
        )
        sys.exit(1)

    step_path = sys.argv[1]

    try:
        print(f"Loading STEP file:\n  {step_path}")

        shape = load_step(step_path)

        print("\nSTEP loaded successfully.")
        print(f"Null shape:  {shape.IsNull()}")
        print(f"Shape type:  {shape.ShapeType()}")
        print(f"Valid B-Rep: {validate_shape(shape)}")

    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)