defmodule TodoHaiku.SyllableCounter do
  @moduledoc """
  Syllable counting using big-phoney microservice as the primary approach.
  """

  @doc """
  Count syllables in text using big-phoney microservice.

  Returns {:ok, syllable_count} on success or {:error, reason} on failure.
  """
  def count_syllables(text) when is_binary(text) do
    TodoHaiku.BigPhoneyClient.count_syllables(text)
  end

  @doc """
  Count syllables in a line of text (for haiku validation).
  """
  def count_syllables_in_line(text) when is_binary(text) do
    case count_syllables(text) do
      {:ok, syllables} -> syllables
      {:error, _reason} -> 0
    end
  end

  @doc """
  Count syllables for multiple words efficiently.

  Takes a list of words and returns a map of word -> syllable count.
  """
  def count_syllables_batch(words) when is_list(words) do
    TodoHaiku.BigPhoneyClient.count_syllables_batch(words)
  end
end
