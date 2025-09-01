-- Stored procedures for post_id_mapping_table operations

-- 1. INSERT/UPSERT stored procedure
CREATE OR REPLACE FUNCTION recsys.upsert_post_id_mapping(
    p_video_id VARCHAR,
    p_publisher_id VARCHAR,
    p_old_post_id VARCHAR,
    p_new_post_id VARCHAR
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO recsys.post_id_mapping_table (video_id, publisher_id, old_post_id, new_post_id)
    VALUES (p_video_id, p_publisher_id, p_old_post_id, p_new_post_id)
    ON CONFLICT (video_id) 
    DO UPDATE SET 
        publisher_id = EXCLUDED.publisher_id,
        old_post_id = EXCLUDED.old_post_id,
        new_post_id = EXCLUDED.new_post_id,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- 2. READ by primary key
CREATE OR REPLACE FUNCTION recsys.get_post_id_mapping(
    p_video_id VARCHAR
)
RETURNS TABLE(
    video_id VARCHAR,
    publisher_id VARCHAR,
    old_post_id VARCHAR,
    new_post_id VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT m.video_id, m.publisher_id, m.old_post_id, m.new_post_id, m.created_at, m.updated_at
    FROM recsys.post_id_mapping_table m
    WHERE m.video_id = p_video_id;
END;
$$ LANGUAGE plpgsql;


-- 3. UPDATE existing mapping
CREATE OR REPLACE FUNCTION recsys.update_post_id_mapping(
    p_video_id VARCHAR,
    p_publisher_id VARCHAR DEFAULT NULL,
    p_old_post_id VARCHAR DEFAULT NULL,
    p_new_post_id VARCHAR DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    rows_affected INTEGER;
BEGIN
    UPDATE recsys.post_id_mapping_table 
    SET 
        publisher_id = COALESCE(p_publisher_id, publisher_id),
        old_post_id = COALESCE(p_old_post_id, old_post_id),
        new_post_id = COALESCE(p_new_post_id, new_post_id),
        updated_at = CURRENT_TIMESTAMP
    WHERE video_id = p_video_id;
    
    GET DIAGNOSTICS rows_affected = ROW_COUNT;
    RETURN rows_affected > 0;
END;
$$ LANGUAGE plpgsql;

-- 4. DELETE mapping
CREATE OR REPLACE FUNCTION recsys.delete_post_id_mapping(
    p_video_id VARCHAR
)
RETURNS BOOLEAN AS $$
DECLARE
    rows_affected INTEGER;
BEGIN
    DELETE FROM recsys.post_id_mapping_table 
    WHERE video_id = p_video_id;
    
    GET DIAGNOSTICS rows_affected = ROW_COUNT;
    RETURN rows_affected > 0;
END;
$$ LANGUAGE plpgsql;


-- 5. Batch insert/upsert function
CREATE OR REPLACE FUNCTION recsys.batch_upsert_post_id_mappings(
    mappings JSONB
)
RETURNS INTEGER AS $$
DECLARE
    mapping JSONB;
    rows_affected INTEGER := 0;
BEGIN
    FOR mapping IN SELECT jsonb_array_elements(mappings)
    LOOP
        INSERT INTO recsys.post_id_mapping_table (video_id, publisher_id, old_post_id, new_post_id)
        VALUES (
            mapping->>'video_id',
            mapping->>'publisher_id',
            mapping->>'old_post_id',
            mapping->>'new_post_id'
        )
        ON CONFLICT (video_id) 
        DO UPDATE SET 
            publisher_id = EXCLUDED.publisher_id,
            old_post_id = EXCLUDED.old_post_id,
            new_post_id = EXCLUDED.new_post_id,
            updated_at = CURRENT_TIMESTAMP;
        
        rows_affected := rows_affected + 1;
    END LOOP;
    
    RETURN rows_affected;
END;
$$ LANGUAGE plpgsql;