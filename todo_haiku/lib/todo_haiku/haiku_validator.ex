defmodule TodoHaiku.HaikuValidator do
  @moduledoc """
  Utilities for validating and generating haikus.
  """

  # Define the haiku pattern (5-7-5 syllables)
  @haiku_pattern [5, 7, 5]

  # Define special cases and exceptions
  @special_cases %{
    "every" => 2,
    "different" => 3,
    "beautiful" => 3,
    "interesting" => 3,
    "experience" => 4,
    "favorite" => 3,
    "family" => 3,
    "evening" => 2,
    "area" => 3,
    "hour" => 1,
    "fire" => 1,
    "poem" => 2,
    "poems" => 2,
    "quiet" => 2,
    "science" => 2,
    "society" => 3,
    "though" => 1,
    "through" => 1,
    "throughout" => 2,
    "wednesday" => 3,
    "forest" => 2,
    "poetry" => 3,
    "haiku" => 2,
    "syllable" => 3,
    "syllables" => 3,
    "deadline" => 2,
    "approaching" => 3,
    # Additional common words with tricky syllable counts
    "actually" => 4,
    "basically" => 3,
    "beautiful" => 3,
    "business" => 2,
    "completely" => 3,
    "definitely" => 4,
    "different" => 3,
    "evening" => 2,
    "every" => 2,
    "everything" => 3,
    "interesting" => 3,
    "literally" => 4,
    "memory" => 3,
    "natural" => 2,
    "generally" => 4,
    "government" => 3,
    "probably" => 3,
    "separately" => 4,
    "several" => 2,
    "vegetable" => 4,
    "being" => 2,
    "create" => 2,
    "going" => 2,
    "power" => 2
  }

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
  Generates a random haiku template to inspire users.
  """
  def generate_template do
    templates = [
      "Morning sun rises\nDew drops glisten on green leaves\nA new day begins",
      "Typing on keyboard\nThoughts flow into characters\nTasks become haikus",
      "Mountain silhouette\nShadows dance across the lake\nPeace in solitude",
      "Deadline approaching\nFingers race across the keys\nWork becomes a blur",
      "Empty task list waits\nIdeas form in my mind\nTime to write them down"
    ]

    Enum.random(templates)
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
    if text == nil or text == "" do
      0
    else
      # Split text into words
      text
      |> String.split(~r/\s+/)
      |> Enum.filter(&(&1 != ""))
      |> Enum.reduce(0, fn word, total -> total + count_syllables_in_word(word) end)
    end
  end

  # Count syllables in a word
  defp count_syllables_in_word(word) do
    # Remove punctuation and convert to lowercase
    word = word |> String.downcase() |> String.replace(~r/[.,;:!?()'"]/u, "")

    # Debug output for the word and its syllable count
    result = case Map.get(@special_cases, word) do
      nil ->
        count = count_by_vowel_groups(word)
        IO.puts("Counting syllables for word: '#{word}' => #{count} (by algorithm)")
        count
      count ->
        IO.puts("Counting syllables for word: '#{word}' => #{count} (from special cases)")
        count
    end

    result
  end

  # Count syllables based on vowel groups
  defp count_by_vowel_groups(word) do
    # Basic vowels
    vowels = "aeiouy"

    # Common diphthongs (count as one syllable)
    diphthongs = ["aa", "ae", "ai", "ao", "au", "ay", "ea", "ee", "ei", "eo", "eu", "ey",
                  "ia", "ie", "ii", "io", "iu", "iy", "oa", "oe", "oi", "oo", "ou", "oy",
                  "ua", "ue", "ui", "uo", "uu", "uy", "ya", "ye", "yi", "yo", "yu"]

    # Return 1 for very short words
    if String.length(word) <= 2 do
      1
    else
      # Handle words with common patterns
      cond do
        # Words ending with "ing"
        String.ends_with?(word, "ing") ->
          if String.match?(word, ~r/[aeiouy]ing$/) do
            # If vowel before "ing", count separately (e.g., "flying" = 2)
            count_without_suffix = count_by_vowel_groups(String.slice(word, 0..-4))
            count_without_suffix + 1
          else
            # Otherwise, count as part of syllable (e.g., "sing" = 1)
            count_without_suffix = count_by_vowel_groups(String.slice(word, 0..-4))
            count_without_suffix
          end

        # Handle -es, -ed endings
        String.ends_with?(word, "es") and String.length(word) > 3 and not String.contains?(vowels, String.at(word, String.length(word) - 3)) ->
          count_without_suffix = count_by_vowel_groups(String.slice(word, 0..-3))
          count_without_suffix

        String.ends_with?(word, "e") and String.length(word) > 2 and not String.contains?(vowels, String.at(word, String.length(word) - 2)) ->
          count_without_suffix = count_by_vowel_groups(String.slice(word, 0..-2))
          count_without_suffix

        String.ends_with?(word, "ed") and String.length(word) > 3 and not String.contains?(vowels, String.at(word, String.length(word) - 3)) ->
          count_without_suffix = count_by_vowel_groups(String.slice(word, 0..-3))
          count_without_suffix

        # Otherwise, count vowel groups
        true ->
          count_vowel_groups(word, vowels, diphthongs)
      end
    end
  end

  # Count vowel groups, with special handling for diphthongs
  defp count_vowel_groups(word, vowels, diphthongs) do
    graphemes = String.graphemes(word)

    # First pass: mark diphthongs
    {marked_word, _} = Enum.reduce(0..(String.length(word) - 2), {graphemes, []}, fn i, {word_acc, diphs} ->
      if i < length(word_acc) - 1 do
        pair = Enum.at(word_acc, i) <> Enum.at(word_acc, i + 1)
        if Enum.member?(diphthongs, pair) and not Enum.member?(diphs, i) do
          # Mark first letter of diphthong with "*" and remove the second
          word_with_marked_diph =
            List.replace_at(word_acc, i, Enum.at(word_acc, i) <> "*")
            |> List.replace_at(i + 1, "")
          {word_with_marked_diph, [i | diphs]}
        else
          {word_acc, diphs}
        end
      else
        {word_acc, diphs}
      end
    end)

    # Count vowel groups
    marked_word = Enum.reject(marked_word, fn x -> x == "" end)

    # Rebuild the word and count vowel groups
    new_word = Enum.join(marked_word, "")

    # Basic counting based on vowel transitions
    {count, prev_is_vowel} =
      if String.length(new_word) > 0 and String.contains?(vowels, String.at(new_word, 0)) do
        {1, true}
      else
        {0, false}
      end

    # Count syllables scanning through the word
    {count, _} =
      Enum.reduce(String.graphemes(new_word) |> Enum.drop(1), {count, prev_is_vowel}, fn char, {acc, prev} ->
        is_vowel = String.contains?(vowels, char) and not String.contains?(char, "*")
        new_count = if is_vowel and not prev, do: acc + 1, else: acc
        {new_count, is_vowel}
      end)

    # Apply final adjustments
    count = apply_ending_rules(word, count, vowels)

    # Ensure at least one syllable
    max(1, count)
  end

  # Apply rules for word endings
  defp apply_ending_rules(word, count, vowels) do
    cond do
      # Handle silent e at the end
      String.length(word) > 2 and String.ends_with?(word, "e") and not String.contains?(vowels, String.at(word, String.length(word) - 2)) ->
        max(1, count - 1)

      # Handle words ending with 'le' where the 'l' is preceded by a consonant
      String.length(word) > 2 and String.ends_with?(word, "le") and not String.contains?(vowels, String.at(word, String.length(word) - 3)) ->
        count + 1

      # Handle words ending with 'ed'
      String.length(word) > 2 and String.ends_with?(word, "ed") ->
        # Only count as a syllable if preceded by t or d
        prev_char = String.at(word, String.length(word) - 3)
        if prev_char != "t" and prev_char != "d" do
          max(1, count - 1)
        else
          count
        end

      true ->
        count
    end
  end
end
