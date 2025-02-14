# Collection

!!! warning 
    Experimental - Syntax or Docs might change

The Collection class is the core data structure in HTRFlow, designed to manage and process document pages in a hierarchical structure. It provides a flexible and intuitive way to handle document analysis tasks, from page segmentation to text recognition.

## Overview

The Collection class maintains a tree structure where:

- Root level contains PageNode objects (individual document pages)
- Pages can contain regions (text blocks, tables, etc.)
- Regions can contain paragraphs or lines of text
- Lines contain individual words

Each node in the tree has associated attributes like position (coordinates), dimensions, and potentially recognized text.

Note that the the Collection underneath consists of three main components that work together:

1. **Collection**: The root container managing document pages and their hierarchy
2. **Result**: Processing outputs that update the Collection's structure
3. **Geometry**: Spatial utilities used throughout the Collection tree

Here's how they interact:
```
Collection
├── Manages PageNodes
│   └── Updated by Results
│       └── Uses Geometry for spatial operations
```

## Basic Usage

Here's a typical workflow using Collection with a pipeline:

```python
from htrflow.pipeline.pipeline import Pipeline
from htrflow.volume.volume import Collection
import yaml

# Create collection from images
collection = Collection(['image1.jpg', 'image2.jpg'])

# Define pipeline configuration
config = yaml.safe_load("""
steps:
- step: Segmentation
  settings:
    model: yolo
    model_settings:
      model: Riksarkivet/yolov9-lines-within-regions-1
- step: TextRecognition
  settings:
    model: TrOCR
    model_settings:
      model: Riksarkivet/trocr-base-handwritten-hist-swe-2
""")


# Process with pipeline
pipe = Pipeline.from_config(config)
collection = pipe.run(collection)

print(collection)
```

output:
```bash
collection tree: # Root
img_h x img_w node (image0) at (origo) # Image 0 (parent)
    └──node0_h x node0_w  (image0_node0) at (px_0, py_0)  # (child)
        ├──node00_h x node00_w (image0_node0_node0) at (px_00, py_00): text0 # (child's child) 
        ├──node01_h x node01_w (image0_node0_node1) at (px_01, py_01): text1
        ├──node02_h x node02_w (image0_node0_node2) at (px_02, py_02): text2
...
img_h x img_w node (image1) at (origo) # Image 1
    └──...
```

## Working with Collection

### Navigation

The Collection class uses intuitive indexing for accessing nodes:

```python
page = collection[0]               # First page
region = collection[0][0]          # First region in first page
line = collection[0][0][0]         # First line in first region
```

or

```python
page = collection[0]            
region = collection[0,0]       
line = collection[0,0,0]    
```

For instance if we have a populated collection class that looked like this:
```bash
collection label: Col_output
collection tree:
2413x1511 node (img) at (0, 0)
    └──2123x1444 node (img_node0) at (54, 218)
        ├──166x965 node (img_node0_node0) at (358, 224): text0
        ├──222x1138 node (img_node0_node1) at (331, 450): text1
        ├──156x1045 node (img_node0_node2) at (437, 702): text2
        ├──119x1191 node (img_node0_node3) at (238, 888): text3
```

Running:
```python
print(col[0])
print(col[0,0])
print(col[0,0,0])
```
 
outputs:
```bash
2413x1511 node (img) at (0, 0)
2123x1444 node (img_node0) at (54, 218)
166x965 node (img_node0_node0) at (358, 224): text0
```

### Node Types

The tree structure consists of different types of nodes that can be identified using these methods:

- `is_region()`: True for nodes containing lines/text blocks
- `is_line()`: True for nodes containing words/text lines
- `is_word()`: True for nodes containing single words

Example:
```python
# Check node types
col[0].is_region()       # True - Page is a region
col[0,0].is_region()     # True - First child is a region
col[0,0,0].is_line()     # True - First grandchild is a line

col[0].is_line()         # False
col[0,0].is_line()       # False
col[0,0,0].is_region()   # False
```


### Traversing the Tree


You can traverse nodes using filters:

```python
# Get specific node types
lines = collection.traverse(filter=lambda node: node.is_line())
regions = collection.traverse(filter=lambda node: node.is_region())
text_nodes = collection.traverse(filter=lambda node: node.contains_text())
```

### Saving and Serialization

The Collection class supports various serialization formats:

```python
# ALTO XML
collection.save(directory="output", serializer="alto")

# Other supported formats:
# - Page XML
# - txt
# - Json
```

## Creating a Collection

```python
from htrflow.volume.volume import Collection

# From individual image files
collection = Collection(['page1.jpg', 'page2.jpg'])

# From a directory
collection = Collection.from_directory('path/to/images')

# From a previously saved collection
collection = Collection.from_pickle('saved_collection.pkl')
```

## Updating Collection (without pipeline)

Collection nodes are updated through model results, with each update potentially modifying the tree structure. Here's a complete example with actual output:


Python: 

=== "1. Create Collection"
    ```python
    # 1. Create collection from images
    collection = Collection(["img.jpg"])
    ```

=== "2. Region Detection"

    ```python
    region_results = dummy_segmentation_model(collection.segments()) # or images()
    collection.update(region_results)
    ```

=== "3. Line Detection"

    ```python
    line_results = dummy_segmentation_model(collection.segments())
    collection.update(line_results)
    ```

=== "4. Text Recognition"

    ```python
    text_results = dummy_text_recognition_model(collection.segments())
    collection.update(text_results)  
    ```


Output:

=== "1. Create Collection"
    ```bash
    Initial tree structure:
    collection label: img
    collection tree:
    5168x6312 node (img) at (0, 0)
    ```

=== "2. Region Detection"
    ```bash
    Tree after region detection:
    collection label: img
    collection tree:
    5168x6312 node (img) at (0, 0)
        ├──1293x1579 node (img_node0) at (3020, 831)
        └──1293x1579 node (img_node1) at (2880, 1376)
    ```


=== "3. Line Detection"
    ```bash
    Tree after line detection:
    collection label: img
    collection tree:
    5168x6312 node (img) at (0, 0)
        ├──1293x1579 node (img_node0) at (3020, 831)
        │   ├──167x395 node (img_node0_node0) at (4204, 1957)
        │   └──323x378 node (img_node0_node1) at (3020, 1086)
        └──1293x1579 node (img_node1) at (2880, 1376)
            ├──323x395 node (img_node1_node0) at (2971, 2208)
            └──323x243 node (img_node1_node1) at (4216, 2272)
    ```

=== "4. Text Recognition"

    ```bash
    Final tree with text:
    collection label: img.jpg
    collection tree:
    5168x6312 node (img) at (0, 0)
        ├──1293x1579 node (img_node0) at (3020, 831)
        │   ├──167x395 node (img_node0_node0) at (4204, 1957): "Magnam sit est ut dolorem consectetur."
        │   └──323x378 node (img_node0_node1) at (3020, 1086): "Dolorem dolore consectetur porro voluptatem eius quaerat dolore."
        └──1293x1579 node (img_node1) at (2880, 1376)
            ├──323x395 node (img_node1_node0) at (2971, 2208): "Sit est velit numquam modi adipisci dolorem ut."
            └──323x243 node (img_node1_node1) at (4216, 2272): "Quiquia quiquia modi modi consectetur sit numquam."
    ```

