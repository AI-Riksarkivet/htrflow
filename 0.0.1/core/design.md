# Htrflow Design

```mermaid
graph TD
    A[htrflow_core] --> B(dummies)
    A --> C(image)
    A --> E(logging)
    A --> F(models)
    A --> G(overlapping_masks)

    A --> J(reading_order)
    A --> K(results)
    A --> L(serialization)
    A --> M(templates)
    A --> N(volume)
    
    
    F --> F3(huggingface)
    F --> F5(openmmlab)
    F --> F7(ultralytics)

    M --> M1(alto)
    M --> M2(page)
```


# Sequence Diagram workflow

The Swedish National Archives introduces a...

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo.

# Data strucutre

```mermaid
graph TD
    A("Image.jpg") --> B("Region")
    B --> C1("Line - Text: 'Lorem non ipsum dolor.'")
    C1 --> D1("Word: 'Lorem'")
    C1 --> D2("Word: 'non'")
    C1 --> D3("Word: 'ipsum'")
    C1 --> D4("Word: 'dolor.'")
    B --> C2("Line - Text: 'Numquam consectetur ut'")
    C2 --> D5("Word: 'Numquam'")
    C2 --> D6("Word: 'consectetur'")
    C2 --> D7("Word: 'ut'")
```
