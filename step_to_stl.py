from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer


step_file = "/home/fatima/Desktop/isf1/isf_first/data/input/bowl.step"
stl_file = "/home/fatima/Desktop/isf1/isf_first/data/input/bowl.stl"

# Read STEP
reader = STEPControl_Reader()

status = reader.ReadFile(step_file)

if status != IFSelect_RetDone:
    raise RuntimeError("Failed to read STEP file")

print("STEP loaded successfully")

# Convert STEP entities into OCC shape
reader.TransferRoots()
shape = reader.OneShape()

if shape.IsNull():
    raise RuntimeError("CAD shape is empty")

print("CAD shape created successfully")

# Triangulate CAD geometry
mesh = BRepMesh_IncrementalMesh(
    shape,
    0.1,    # linear deflection
    False,  # relative deflection
    0.5,    # angular deflection
    True    # parallel
)

mesh.Perform()

if not mesh.IsDone():
    raise RuntimeError("Triangulation failed")

print("Triangulation completed")

# Write STL
writer = StlAPI_Writer()
writer.Write(shape, stl_file)

print("STL saved to:")
print(stl_file)