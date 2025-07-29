defmodule TodoHaiku.SyllableCounter do
  @moduledoc """
  Syllable counting using hybrid ONNX + dictionary approach.
  Real-time ML inference for partial words, dictionary accuracy for complete words.
  """

  @doc """
  Count syllables in text using hybrid approach.

  Returns {:ok, syllable_count} on success or {:error, reason} on failure.
  """
  def count_syllables(text) when is_binary(text) do
    TodoHaiku.AxonSyllableCounter.count_syllables(text)
  end

  @doc """
  Count syllables in a line of text (for haiku validation).
  """
  def count_syllables_in_line(text) when is_binary(text) do
    TodoHaiku.AxonSyllableCounter.count_syllables_in_line(text)
  end

  @doc """
  Count syllables for multiple words efficiently.

  Takes a list of words and returns a map of word -> syllable count.
  """
  def count_syllables_batch(words) when is_list(words) do
    TodoHaiku.AxonSyllableCounter.count_syllables_batch(words)
  end

  @doc """
  Count syllables for haiku lines (returns structured data for validation).
  """
  def count_syllables_haiku(content) when is_binary(content) do
    TodoHaiku.AxonSyllableCounter.count_syllables_haiku(content)
  end
end
