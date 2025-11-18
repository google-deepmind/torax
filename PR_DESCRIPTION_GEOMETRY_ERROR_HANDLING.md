# Improve geometry file loading error handling

**Addresses:** #1662

## Summary

Adds comprehensive error handling for geometry file loading (CHEASE, FBT, EQDSK) with helpful error messages, file suggestions, and pre-parsing validation.

## Problem

Geometry file loading errors were unhelpful:
- Generic `FileNotFoundError` without context
- Deep library stack traces obscuring the real issue
- No suggestions when files weren't found
- No validation before parsing

## Solution

### 1. Custom Exception Hierarchy

```python
GeometryFileError (base)
├── GeometryFileNotFoundError  # with similar file suggestions
├── GeometryFileFormatError     # with specific format details
├── GeometryFilePermissionError
├── GeometryFileEmptyError
└── GeometryDataValidationError # for future use
```

### 2. Modular Error Handling

Created `geometry_error_handling.py` module with:
- `find_similar_files()` - Uses difflib for filename suggestions
- `validate_file_access()` - Pre-parsing validation

Keeps `geometry_loader.py` focused on core loading logic.

### 3. Better Error Messages

**Before:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'geometry.txt'
```

**After:**
```
GeometryFileNotFoundError: Geometry file not found: 'geometry.txt'
  Searched in directory: /path/to/geo

  Did you mean one of these?
    - geometry_file.txt
    - geometry_data.txt

  Action: Check the file path and ensure the file exists.
```

### 4. Format-Specific Validation

- **CHEASE**: Header, column count, data type validation
- **FBT**: MATLAB structure validation
- **EQDSK**: COCOS parameter validation, required fields

### 5. Logging Integration

- INFO logs for success
- ERROR logs for failures
- EXCEPTION logs with tracebacks

## Changes

**New Files:**
- `geometry_errors.py` - Exception classes
- `geometry_error_handling.py` - Helper functions
- `tests/geometry_loader_test.py` - Comprehensive tests (15+ cases)

**Modified Files:**
- `geometry_loader.py` - Enhanced with error handling

**Stats:** 3 files changed, 600+ insertions, 50+ deletions

## Testing

✅ File not found with suggestions
✅ Permission errors
✅ Empty file detection
✅ Format errors (CHEASE, FBT, EQDSK)
✅ Valid file loading (regression)

## Backward Compatibility

✅ **Fully backward compatible**
- No API changes
- Only improves error messages on failures
- No performance impact on success path

## Code Quality

✅ Modular design with separate error handling module
✅ Comprehensive docstrings
✅ Type hints throughout
✅ Exception chaining preserves context
✅ No new dependencies

## Impact

- **New users:** Easier to debug configuration issues
- **Researchers:** Faster troubleshooting
- **CI/CD:** Better error messages in logs
- **Support:** Clearer error reports

---

**Ready for review!** 🚀
