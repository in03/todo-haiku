-- This seed file will be executed when running 'supabase db reset'
-- It's useful for testing and development

-- Insert some example todos (these will be associated with the test user)
INSERT INTO todos (title, content, user_id, is_completed)
VALUES 
  (
    'Morning Routine', 
    'Wake with the sunrise
Stretch and breathe the morning air
Day begins anew', 
    '00000000-0000-0000-0000-000000000000', 
    false
  ),
  (
    'Project Deadline', 
    'Code flows like water
Deadline approaches swiftly
Fingers type faster', 
    '00000000-0000-0000-0000-000000000000', 
    false
  ),
  (
    'Evening Meditation', 
    'Mind becomes still pond
Thoughts settle like fallen leaves
Peace fills the silence', 
    '00000000-0000-0000-0000-000000000000', 
    true
  );

-- Note: The user_id above is a placeholder. In a real environment, 
-- you would replace it with an actual user ID after creating a test user.
