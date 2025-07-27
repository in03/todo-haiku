defmodule TodoHaiku.HaikuValidator do
  @moduledoc """
  Utilities for validating and generating haikus.
  """

  # Define the haiku pattern (5-7-5 syllables)
  @haiku_pattern [5, 7, 5]



  @doc """
  Validates if the given text follows the haiku pattern (5-7-5 syllables).
  Returns a tuple with {is_valid, syllable_counts, feedback}
  """
  def validate_haiku(text) do
    # Debug the input
    IO.puts("Validating haiku: #{inspect(text)}")

    # Split text into lines and remove empty lines
    lines = text |> String.split(~r/\r?\n/) |> Enum.filter(&(String.trim(&1) != ""))

    # Debug the split lines
    IO.puts("Split into lines: #{inspect(lines)}")

    # Check if we have exactly 3 lines
    if length(lines) != 3 do
      {
        false,
        Enum.map(lines, &count_syllables/1),
        "A haiku needs exactly three lines."
      }
    else
      # Count syllables for each line
      syllable_counts = Enum.map(lines, &count_syllables/1)

      # Debug syllable counts
      IO.puts("Syllable counts: #{inspect(syllable_counts)}")

      # Check if syllable counts match the haiku pattern
      is_valid = Enum.zip(syllable_counts, @haiku_pattern)
                 |> Enum.all?(fn {actual, expected} -> actual == expected end)

      # Generate feedback based on validation
      feedback = generate_feedback(is_valid, syllable_counts)

      {is_valid, syllable_counts, feedback}
    end
  end

  @doc """
  Generates a template haiku to help users get started.
  """
  def generate_template do
    # Choose a random example from our list
    examples = [
      {"Do Laundry",
       "High piles of laundry\nGetting so tired of this\nWhen will it all end?"},

      {"Morning Exercise",
       "Early morning run\nFeet pounding on the pavement\nStrength builds with each step"},

      {"Study Session",
       "Books spread on the desk\nKnowledge flows through fingertips\nMind grows like a tree"},

      {"Self Care Evening",
       "Candles flicker soft\nRelaxation washes through\nTime just for myself"},

      {"Grocery Shopping",
       "Empty pantry calls\nWheels squeak along tile floors\nFridge now overflows"}
    ]

    # Pick a random example
    {_title, haiku} = Enum.random(examples)

    # Return the haiku text
    haiku
  end

  # Generate feedback based on validation results
  defp generate_feedback(true, _) do
    [
      "Perfect haiku! You're a natural poet.",
      "Wow, a real poet!",
      "Your haiku flows like a gentle stream.",
      "Basho would be proud!",
      "A moment captured in seventeen syllables."
    ]
    |> Enum.random()
  end

  defp generate_feedback(false, syllable_counts) do
    # Check which lines don't match the pattern
    invalid_lines = Enum.zip(syllable_counts, @haiku_pattern)
                    |> Enum.with_index(1)
                    |> Enum.filter(fn {{actual, expected}, _line} -> actual != expected end)
                    |> Enum.map(fn {{actual, expected}, line} -> {line, expected, actual, expected - actual} end)

    # Check total count
    total_syllables = Enum.sum(syllable_counts)
    ideal_syllables = Enum.sum(@haiku_pattern)

    cond do
      # Missing lines
      length(syllable_counts) < 3 ->
        "A haiku needs exactly three lines. You have #{length(syllable_counts)}."

      # First check specific line issues
      length(invalid_lines) == 1 ->
        {line, expected, actual, diff} = hd(invalid_lines)
        if diff > 0 do
          "Line #{line} needs #{diff} more #{if diff == 1, do: "syllable", else: "syllables"} (has #{actual}, needs #{expected})."
        else
          "Line #{line} has #{abs(diff)} too many #{if abs(diff) == 1, do: "syllable", else: "syllables"} (has #{actual}, needs #{expected})."
        end

      # Then check overall progress
      total_syllables < ideal_syllables ->
        "Your haiku has #{total_syllables} syllables. You need #{ideal_syllables - total_syllables} more for a perfect 5-7-5 pattern."

      total_syllables > ideal_syllables ->
        "Your haiku has #{total_syllables} syllables. That's #{total_syllables - ideal_syllables} too many for a perfect 5-7-5 pattern."

      # General encouragement
      true ->
        [
          "Not quite a haiku, but it's a start!",
          "Oops, missing a few syllables.",
          "Check your syllable count on each line.",
          "Almost there! Keep adjusting your words.",
          "A valiant attempt at poetry. Keep refining!"
        ]
        |> Enum.random()
    end
  end

  # Count syllables in a line of text
  defp count_syllables(text) do
    TodoHaiku.SyllableCounter.count_syllables_in_line(text)
  end


end
