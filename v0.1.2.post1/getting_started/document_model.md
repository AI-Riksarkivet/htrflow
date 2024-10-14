# Document model
This page will be updated to explain HTRflow's document model.

## Collection
A `Collection` is an abstraction over one or several input pages:

```
Collection
├── Page
└── Page
    ├── node
    |   ├── node
    |   └── node
    └── node
        ├── node
        └── node
```
