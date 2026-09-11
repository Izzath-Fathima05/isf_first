"""
General CAD-native face classification + flattening
======================================================

Unlike bowl_patches (single continuous revolved surface -> cone
patches), a general part like a bracket is a B-Rep made of several
DIFFERENT face types stitched together. The correct workflow is:

1. Load the real B-Rep (not a triangulated mesh) via STEPControl_Reader
   -- this preserves exact surface types (Plane, Cylinder, Cone, ...),
   which a mesh (trimesh/STL) throws away.
2. Classify EVERY face by its exact surface type.
3. Flatten each face using the formula appropriate to ITS type:
   - Plane:    already flat -- no unrolling needed at all.
   - Cylinder: unrolls into a RECTANGLE (width = R*angle, height = axial
     extent) -- note this is a DIFFERENT formula from a cone; a cylinder
     is not just "a cone with zero half-angle", it needs its own
     parametrization.
   - Cone:     unrolls into an annular sector (see cad_native_patches.py).
   - Anything else (general/doubly-curved, e.g. a real bowl wall): NOT
     exactly flattenable -- falls back to the numerical/approximate
     approach (patch_extraction.py + backward_solve.py).
"""

from OCP.BRepAdaptor import BRepAdaptor_Surface  # noqa: I001  # type: ignore[reportMissingImports]
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
import OCP.TopoDS
from OCP.gp import gp_Pnt


def load_step_native(path: str) -> OCP.TopoDS.TopoDS_Shape:
    """Loads a STEP file as a real B-Rep (exact surface types preserved),
    NOT a triangulated mesh -- required to know which faces are exactly
    planar/cylindrical/conical vs. genuinely doubly-curved."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path}")
    reader.TransferRoots()
    return reader.OneShape()


def classify_faces(shape: OCP.TopoDS.TopoDS_Shape) -> dict:
    """Walks every face in the shape and classifies it by exact
    surface type. Returns a dict of lists, keyed by type name."""
    result = {"Plane": [], "Cylinder": [], "Cone": [], "Other": []}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = OCP.TopoDS.TopoDS.Face_s(explorer.Current())
        adaptor = BRepAdaptor_Surface(face)
        t = adaptor.GetType()
        if t == GeomAbs_Plane:
            result["Plane"].append(face)
        elif t == GeomAbs_Cylinder:
            result["Cylinder"].append(face)
        elif t == GeomAbs_Cone:
            result["Cone"].append(face)
        else:
            result["Other"].append(face)
        explorer.Next()
    return result


def flatten_cylindrical_face(face, x_offset=0.0, y_offset=0.0):
    """
    Unrolls a cylindrical face into its exact flat rectangle.
    Geom_CylindricalSurface parametrization: P(u,v) = center + v*axis +
    R*(cos(u)*xdir + sin(u)*ydir) -- so u (angle) and v (axial position)
    ALREADY unroll linearly: flat_x = R*u, flat_y = v. This is a
    DIFFERENT (simpler) formula from the cone case -- a cylinder is the
    R->infinity limit of a cone's slant, not a half-angle=0 special case
    of the cone formula.
    """
    adaptor = BRepAdaptor_Surface(face)
    cyl = adaptor.Cylinder()
    R = cyl.Radius()
    u1, u2 = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v1, v2 = adaptor.FirstVParameter(), adaptor.LastVParameter()

    width = R * (u2 - u1)   # arc length
    height = v2 - v1        # axial extent

    p1 = gp_Pnt(x_offset, y_offset, 0)
    p2 = gp_Pnt(x_offset + width, y_offset, 0)
    p3 = gp_Pnt(x_offset + width, y_offset + height, 0)
    p4 = gp_Pnt(x_offset, y_offset + height, 0)

    wire = BRepBuilderAPI_MakeWire(
        BRepBuilderAPI_MakeEdge(p1, p2).Edge(),
        BRepBuilderAPI_MakeEdge(p2, p3).Edge(),
        BRepBuilderAPI_MakeEdge(p3, p4).Edge(),
        BRepBuilderAPI_MakeEdge(p4, p1).Edge(),
    ).Wire()
    flat_face = BRepBuilderAPI_MakeFace(wire).Face()
    return flat_face, {"type": "Cylinder", "radius": R, "width": width, "height": height}


def get_planar_face_extent(face):
    """Planar faces are already flat -- no unrolling needed. Just
    report their bounding info for layout purposes."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box = Bnd_Box()
    BRepBndLib.Add_s(face, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return {"type": "Plane", "bbox": (xmin, ymin, zmin, xmax, ymax, zmax)}


def export_step(shapes: list, path: str):
    writer = STEPControl_Writer()
    for s in shapes:
        writer.Transfer(s, STEPControl_AsIs)
    if writer.Write(path) != IFSelect_RetDone:
        raise RuntimeError("STEP export failed")