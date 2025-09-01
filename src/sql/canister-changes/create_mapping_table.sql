-- Create the recsys schema
CREATE SCHEMA IF NOT EXISTS
  recsys;


CREATE TABLE
  recsys.post_id_mapping_table (
    video_id VARCHAR NOT NULL,
    publisher_id VARCHAR NOT NULL,
    old_post_id VARCHAR NOT NULL,
    new_post_id VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Primary key constraint
    PRIMARY KEY (video_id)
  );


-- Create indexes separately (PostgreSQL syntax)
CREATE INDEX idx_video_id ON recsys.post_id_mapping_table (video_id);


CREATE INDEX idx_publisher_id ON recsys.post_id_mapping_table (publisher_id);


CREATE INDEX idx_old_post_id ON recsys.post_id_mapping_table (old_post_id);


CREATE INDEX idx_new_post_id ON recsys.post_id_mapping_table (new_post_id);


-- Add a trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION
  update_updated_at_column () RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
RETURN
  NEW;
END;
$$LANGUAGE plpgsql;


CREATE TRIGGER update_post_id_mapping_updated_at BEFORE
UPDATE
  ON recsys.post_id_mapping_table
FOR
  EACH ROW EXECUTE FUNCTION update_updated_at_column ();


-- Optional: Add comments for documentation
COMMENT ON TABLE recsys.post_id_mapping_table IS 'Mapping table for post ID transitions';


COMMENT ON COLUMN recsys.post_id_mapping_table.video_id IS 'Video identifier (32 character hash)';


COMMENT ON COLUMN recsys.post_id_mapping_table.publisher_id IS 'Publisher identifier (hyphenated format)';


COMMENT ON COLUMN recsys.post_id_mapping_table.old_post_id IS 'Original post identifier';


COMMENT ON COLUMN recsys.post_id_mapping_table.new_post_id IS 'New post identifier';


COMMENT ON COLUMN recsys.post_id_mapping_table.created_at IS 'Record creation timestamp';


COMMENT ON COLUMN recsys.post_id_mapping_table.updated_at IS 'Record last update timestamp';
