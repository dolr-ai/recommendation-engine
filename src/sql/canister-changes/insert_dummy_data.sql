-- Insert dummy data for testing post_id_mapping_table using stored procedures

-- Test 1: Insert first mapping using upsert procedure
SELECT recsys.upsert_post_id_mapping(
    'test-abc123def456ghi789jkl012mno3',
    'test-bkyz2-fmaaa-aaaah-qaaaq-cai',
    'test-old_post_123',
    'test-new_post_456'
);

-- Test 2: Insert second mapping
SELECT recsys.upsert_post_id_mapping(
    'test-xyz789abc012def345ghi678jkl9',
    'test-rdmx6-jaaaa-aaaah-qcaiq-cai',
    'test-old_post_789',
    'test-new_post_012'
);

-- Test 3: Insert third mapping with different publisher
SELECT recsys.upsert_post_id_mapping(
    'test-pqr456stu789vwx012yzab345cde',
    'test-be2us-64aaa-aaaah-qadaq-cai',
    'test-legacy_post_001',
    'test-modern_post_001'
);

-- Test 4: Insert fourth mapping
SELECT recsys.upsert_post_id_mapping(
    'test-mno234pqr567stu890vwx123yza4',
    'test-bkyz2-fmaaa-aaaah-qaaaq-cai',
    'test-old_video_content_42',
    'test-new_video_content_42'
);

-- Test 5: Update existing mapping (test upsert functionality)
SELECT recsys.upsert_post_id_mapping(
    'test-abc123def456ghi789jkl012mno3',
    'test-updated-publisher-id-test',
    'test-updated_old_post_123',
    'test-updated_new_post_456'
);

-- Test 6: Insert mapping with longer post IDs
SELECT recsys.upsert_post_id_mapping(
    'test-def678ghi901jkl234mno567pqr8',
    'test-rrkah-fqaaa-aaaah-qaaaq-cai',
    'test-very_long_old_post_identifier_2023',
    'test-very_long_new_post_identifier_2024'
);

-- Verify all inserts worked by reading the data
-- Read by primary key
SELECT 'Test Read by PK:' as test_name, * FROM recsys.get_post_id_mapping('test-abc123def456ghi789jkl012mno3');

-- Read by old post ID
SELECT 'Test Read by Old Post ID:' as test_name, * FROM recsys.get_mapping_by_old_post_id('test-old_post_789');

-- Read by new post ID  
SELECT 'Test Read by New Post ID:' as test_name, * FROM recsys.get_mapping_by_new_post_id('test-new_post_012');

-- Read by publisher
SELECT 'Test Read by Publisher:' as test_name, * FROM recsys.get_mappings_by_publisher('test-bkyz2-fmaaa-aaaah-qaaaq-cai', 10, 0);

-- Test update function
SELECT 'Test Update Function:' as test_name, recsys.update_post_id_mapping(
    'test-mno234pqr567stu890vwx123yza4',
    'test-updated-publisher-via-update-func',
    'test-updated_via_update_func_old',
    'test-updated_via_update_func_new'
) as update_success;

-- Test batch upsert with JSON
SELECT 'Test Batch Upsert:' as test_name, recsys.batch_upsert_post_id_mappings('[
    {
        "video_id": "test-batch001test123456789012345",
        "publisher_id": "test-batch-test-pub-001",
        "old_post_id": "test-batch_old_001", 
        "new_post_id": "test-batch_new_001"
    },
    {
        "video_id": "test-batch002test123456789012346",
        "publisher_id": "test-batch-test-pub-002",
        "old_post_id": "test-batch_old_002",
        "new_post_id": "test-batch_new_002"
    }
]'::jsonb) as batch_rows_affected;

-- Final verification: Show all data in table
SELECT 'All Mappings:' as summary, video_id, publisher_id, old_post_id, new_post_id, created_at, updated_at 
FROM recsys.post_id_mapping_table 
ORDER BY created_at DESC;