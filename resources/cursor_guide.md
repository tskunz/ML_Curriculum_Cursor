# Cursor IDE Guide for Machine Learning Development

## Key Features and Shortcuts

### Navigation
- `Ctrl+P` or `Cmd+P`: Quick file search
- `Ctrl+Shift+P` or `Cmd+Shift+P`: Command palette
- `Ctrl+B` or `Cmd+B`: Toggle sidebar
- `Alt+←/→`: Navigate back/forward
- `Ctrl+G` or `Cmd+G`: Go to line

### Code Editing
- `Alt+↑/↓`: Move line up/down
- `Ctrl+/` or `Cmd+/`: Toggle line comment
- `Ctrl+Space`: Trigger suggestions
- `Ctrl+Shift+K` or `Cmd+Shift+K`: Delete line
- `Alt+Click`: Add multiple cursors

### AI Features
- `Ctrl+K`: Chat with AI assistant
- `Ctrl+I`: Inline AI completion
- `Ctrl+L`: AI explain code
- `Ctrl+;`: AI fix error

### Terminal
- `` Ctrl+` ``: Toggle terminal
- `Ctrl+Shift+5`: Split terminal
- `Alt+1/2/3`: Switch between terminal tabs

## Best Practices for Machine Learning Development in Cursor

### 1. Project Organization
```
project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── tests/
└── README.md
```

### 2. Version Control Integration
- Use Git integration with `Ctrl+Shift+G`
- Stage changes with `+` icon
- Commit with message
- Push/pull with sidebar buttons

### 3. Jupyter Notebook Integration
- Open `.ipynb` files directly
- Run cells with `Shift+Enter`
- Add cells with `B` (below) or `A` (above)
- Change cell type with `Y` (code) or `M` (markdown)

### 4. Debugging
- Set breakpoints by clicking left of line numbers
- Start debugging with F5
- Step over with F10
- Step into with F11
- Variables view in debug sidebar

### 5. AI Assistant Tips
- Be specific in your queries
- Use AI to explain complex code
- Get suggestions for optimizations
- Ask for documentation help

### 6. Code Quality
- Use built-in linter
- Format code with `Alt+Shift+F`
- Check problems panel regularly
- Use AI to suggest improvements

## Tips for Machine Learning Workflow

1. **Data Loading and Preprocessing**
```python
# Instead of Pandas, use pure Python or NumPy
import numpy as np

# Load CSV data
def load_csv(filepath):
    data = []
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            data.append(line.strip().split(','))
    return np.array(data)

# Example usage
data = load_csv('data.csv')
```

2. **Data Manipulation**
```python
# Instead of Pandas DataFrame operations
def filter_data(data, column_idx, condition):
    return data[np.where(data[:, column_idx] == condition)]

def group_by(data, column_idx):
    unique_values = np.unique(data[:, column_idx])
    groups = {val: data[data[:, column_idx] == val] for val in unique_values}
    return groups
```

3. **Visualization**
```python
import matplotlib.pyplot as plt

# Create clear, labeled visualizations
def plot_distribution(data, column_idx, title):
    plt.figure(figsize=(10, 6))
    plt.hist(data[:, column_idx].astype(float))
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
```

Remember to use Cursor's AI features to help understand code, debug issues, and get suggestions for improvements! 