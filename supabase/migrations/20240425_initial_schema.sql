-- Create todos table
CREATE TABLE IF NOT EXISTS todos (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  user_id UUID NOT NULL REFERENCES auth.users(id),
  is_completed BOOLEAN DEFAULT FALSE
);

-- Enable Row Level Security
ALTER TABLE todos ENABLE ROW LEVEL SECURITY;

-- Create policy for users to see only their own todos
CREATE POLICY "Users can only access their own todos" 
  ON todos FOR ALL 
  USING (auth.uid() = user_id);

-- Create policy for users to insert their own todos
CREATE POLICY "Users can insert their own todos" 
  ON todos FOR INSERT 
  WITH CHECK (auth.uid() = user_id);

-- Create policy for users to update their own todos
CREATE POLICY "Users can update their own todos" 
  ON todos FOR UPDATE 
  USING (auth.uid() = user_id);

-- Create policy for users to delete their own todos
CREATE POLICY "Users can delete their own todos" 
  ON todos FOR DELETE 
  USING (auth.uid() = user_id);
