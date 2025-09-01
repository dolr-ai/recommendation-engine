# Post ID Mapping Table - Database Schema and Procedures

This folder contains the database schema and stored procedures for managing post ID mappings in the YRAL recommendation system.

## Files

- **`create_mapping_table.sql`** - Creates the `recsys.post_id_mapping_table` with schema, indexes, triggers, and comments
- **`post_id_mapping_procedures.sql`** - Stored procedures for CRUD operations
- **`insert_dummy_data.sql`** - Test data and validation queries using stored procedures

## Table Schema

```sql
recsys.post_id_mapping_table (
    video_id VARCHAR NOT NULL PRIMARY KEY,
    publisher_id VARCHAR NOT NULL,
    old_post_id VARCHAR NOT NULL,
    new_post_id VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Indexes:** `video_id`, `publisher_id`, `old_post_id`, `new_post_id`  
**Trigger:** Auto-updates `updated_at` on row modifications

## Stored Procedures

All operations are **video_id based** for simplicity and performance.

### 1. Insert/Update Operations

#### `recsys.upsert_post_id_mapping(video_id, publisher_id, old_post_id, new_post_id)`
- **Purpose:** Insert new mapping or update existing one based on `video_id`
- **Returns:** VOID
- **Example:**
```sql
SELECT recsys.upsert_post_id_mapping(
    'abc123def456',
    'publisher-123',
    'old_post_456',
    'new_post_789'
);
```

### 2. Read Operations

#### `recsys.get_post_id_mapping(video_id)`
- **Purpose:** Get mapping by video ID (primary key)
- **Returns:** Table with single row or empty
- **Example:**
```sql
SELECT * FROM recsys.get_post_id_mapping('abc123def456');
```

### 3. Update Operations

#### `recsys.update_post_id_mapping(video_id, publisher_id?, old_post_id?, new_post_id?)`
- **Purpose:** Update existing mapping (optional parameters)
- **Returns:** BOOLEAN (true if row was updated)
- **Example:**
```sql
SELECT recsys.update_post_id_mapping(
    'abc123def456',
    'new-publisher-id',  -- Update publisher
    NULL,                -- Keep old_post_id unchanged
    'updated_new_post'   -- Update new_post_id
);
```

### 4. Delete Operations

#### `recsys.delete_post_id_mapping(video_id)`
- **Purpose:** Delete mapping by video ID
- **Returns:** BOOLEAN (true if row was deleted)
- **Example:**
```sql
SELECT recsys.delete_post_id_mapping('abc123def456');
```

### 5. Batch Operations

#### `recsys.batch_upsert_post_id_mappings(mappings_json)`
- **Purpose:** Batch insert/update multiple mappings from JSON array
- **Returns:** INTEGER (number of rows processed)
- **Example:**
```sql
SELECT recsys.batch_upsert_post_id_mappings('[
    {
        "video_id": "vid1",
        "publisher_id": "pub1", 
        "old_post_id": "old1",
        "new_post_id": "new1"
    },
    {
        "video_id": "vid2",
        "publisher_id": "pub2",
        "old_post_id": "old2", 
        "new_post_id": "new2"
    }
]'::jsonb);
```

## Usage Examples

### Basic CRUD Operations
```sql
-- Insert/Update
SELECT recsys.upsert_post_id_mapping('vid123', 'pub456', 'old789', 'new012');

-- Read by video ID
SELECT * FROM recsys.get_post_id_mapping('vid123');

-- Update specific fields
SELECT recsys.update_post_id_mapping('vid123', NULL, 'updated_old', NULL);

-- Delete by video ID
SELECT recsys.delete_post_id_mapping('vid123');
```

### Batch Operations
```sql
-- Batch insert multiple mappings
SELECT recsys.batch_upsert_post_id_mappings('[
    {"video_id": "v1", "publisher_id": "p1", "old_post_id": "o1", "new_post_id": "n1"},
    {"video_id": "v2", "publisher_id": "p2", "old_post_id": "o2", "new_post_id": "n2"}
]'::jsonb);
```

## Testing

Run `insert_dummy_data.sql` to:
1. Insert test data with `test-` prefixes
2. Validate all stored procedures
3. Verify read operations work correctly

Clean up test data:
```sql
DELETE FROM recsys.post_id_mapping_table WHERE video_id LIKE 'test-%';
```

## Deployment Order

1. Run `create_mapping_table.sql` - Creates schema, table, indexes, triggers
2. Run `post_id_mapping_procedures.sql` - Creates all stored procedures  
3. Run `insert_dummy_data.sql` - Optional testing and validation