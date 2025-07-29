defmodule TodoHaiku.OrtexSyllableCounter do
  @moduledoc """
  Hybrid syllable counting using Ortex for real-time ML inference
  and CMU dictionary for accuracy on complete words.

  Strategy:
  - Real-time typing: Fast ONNX model inference for immediate feedback on partial words
  - Complete words: Dictionary validation for 100% accuracy on known words
  - Unknown complete words: ML model fallback
  """

  use GenServer
  require Logger

  # Character alphabet from model metadata
  @alphabet ["'", "-", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
  @max_word_length 18
  @alphabet_size 28

  defstruct [:model, :char_to_index, :cmu_dict]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Count syllables in text using hybrid approach.
  Real-time: Fast ML inference for immediate feedback
  Complete words: Dictionary validation for accuracy
  """
  def count_syllables(text) when is_binary(text) do
    GenServer.call(__MODULE__, {:count_syllables, text})
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
  Count syllables for haiku lines (returns structured data for validation).
  """
  def count_syllables_haiku(content) when is_binary(content) do
    lines = content
            |> String.split(~r/\r?\n/)
            |> Enum.filter(&(String.trim(&1) != ""))

    case lines do
      [] ->
        {:ok, %{"lines" => []}}

      lines ->
        results = Enum.map(lines, fn line ->
          case count_syllables(line) do
            {:ok, syllables} -> %{"text" => line, "syllables" => syllables}
            {:error, _} -> %{"text" => line, "syllables" => 0}
          end
        end)

        {:ok, %{"lines" => results}}
    end
  end

  @doc """
  Count syllables for multiple words efficiently.
  """
  def count_syllables_batch(words) when is_list(words) do
    GenServer.call(__MODULE__, {:count_syllables_batch, words})
  end

  # GenServer callbacks

  @impl true
  def init(_opts) do
    case load_models() do
      {:ok, state} ->
        Logger.info("OrtexSyllableCounter: Ortex ONNX model and CMU dictionary loaded successfully")
        {:ok, state}

      {:error, reason} ->
        Logger.error("OrtexSyllableCounter: Failed to load models: #{inspect(reason)}")
        {:stop, reason}
    end
  end

  @impl true
  def handle_call({:count_syllables, text}, _from, state) do
    result = predict_syllables_hybrid(text, state)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:count_syllables_batch, words}, _from, state) do
    results = Enum.reduce(words, %{}, fn word, acc ->
      case predict_syllables_hybrid(word, state) do
        {:ok, syllables} -> Map.put(acc, word, syllables)
        {:error, _} -> Map.put(acc, word, 0)
      end
    end)

    {:reply, {:ok, results}, state}
  end

  # Private functions

  defp load_models do
    with {:ok, onnx_state} <- load_onnx_model(),
         {:ok, cmu_dict} <- load_cmu_dictionary() do

      state = Map.merge(onnx_state, %{cmu_dict: cmu_dict})
      {:ok, state}
    else
      {:error, onnx_error} ->
        Logger.warning("ONNX model failed to load: #{inspect(onnx_error)}")
        # Try to continue with just dictionary
        case load_cmu_dictionary() do
          {:ok, cmu_dict} ->
            Logger.info("Continuing with dictionary-only mode")
            {:ok, %{model: nil, char_to_index: nil, cmu_dict: cmu_dict}}
          {:error, dict_error} ->
            {:error, "Both ONNX and dictionary failed: #{inspect({onnx_error, dict_error})}"}
        end
    end
  end

  defp load_onnx_model do
    model_path = "syllable_model.onnx"

    if File.exists?(model_path) do
      try do
        model = Ortex.load(model_path)

        # Create character to index mapping
        char_to_index = @alphabet
                       |> Enum.with_index()
                       |> Map.new()

        state = %{
          model: model,
          char_to_index: char_to_index
        }

        {:ok, state}
      rescue
        error ->
          {:error, "Failed to load ONNX model: #{inspect(error)}"}
      end
    else
      {:error, "Model file not found at #{model_path}"}
    end
  end

  defp load_cmu_dictionary do
    dict_path = "cmudict.dict"

    if File.exists?(dict_path) do
      try do
        cmu_dict = dict_path
                   |> File.stream!()
                   |> Stream.map(&String.trim/1)
                   |> Stream.reject(&String.starts_with?(&1, ";;;"))  # Skip comments
                   |> Stream.map(&parse_cmu_line/1)
                   |> Stream.reject(&is_nil/1)
                   |> Enum.into(%{})

        {:ok, cmu_dict}
      rescue
        error ->
          {:error, "Failed to parse CMU dictionary: #{inspect(error)}"}
      end
    else
      {:error, "CMU dictionary file not found at #{dict_path}"}
    end
  end

  defp parse_cmu_line(line) do
    case String.split(line, "  ", parts: 2) do
      [word, phonemes] ->
        # Count stress markers (0, 1, 2) which indicate syllables
        syllable_count = phonemes
                        |> String.graphemes()
                        |> Enum.count(&(&1 in ["0", "1", "2"]))

        # Normalize word (remove stress indicators like (1), (2), etc.)
        normalized_word = word
                         |> String.downcase()
                         |> String.replace(~r/\(\d+\)$/, "")

        {normalized_word, syllable_count}

      _ ->
        nil
    end
  end

  defp predict_syllables_hybrid(text, state) do
    try do
      words = extract_words_with_boundaries(text)

      if Enum.empty?(words) do
        {:ok, 0}
      else
        total_syllables = Enum.reduce(words, 0, fn {word, is_complete}, acc ->
          syllables = if is_complete do
            # Complete word: Use dictionary first, ML fallback
            predict_complete_word(word, state)
          else
            # Partial word: Use fast ML inference
            predict_partial_word(word, state)
          end
          acc + syllables
        end)

        {:ok, total_syllables}
      end
    rescue
      error ->
        Logger.warning("Error predicting syllables for '#{text}': #{inspect(error)}")
        {:error, "Prediction failed"}
    end
  end

  defp extract_words_with_boundaries(text) do
    # Split text and identify whether each word is "complete" (followed by space/punctuation)
    # vs "partial" (being actively typed)

    # Simple heuristic: if text ends with space or punctuation, all words are complete
    # If text ends with a letter, the last word is partial
    cleaned_text = text
                   |> String.downcase()
                   |> String.replace(~r/[^a-z\s'-]/, " ")

    words = cleaned_text
            |> String.split(~r/\s+/)
            |> Enum.filter(&(String.trim(&1) != ""))

    case words do
      [] -> []
      words_list ->
        # Check if the original text ends with whitespace or punctuation
        text_ends_complete = String.match?(text, ~r/[\s\p{P}]$/)

        if text_ends_complete do
          # All words are complete
          Enum.map(words_list, &{&1, true})
        else
          # Last word is partial, rest are complete
          {complete_words, [last_word]} = Enum.split(words_list, -1)
          complete_words_marked = Enum.map(complete_words, &{&1, true})
          complete_words_marked ++ [{last_word, false}]
        end
    end
  end

  defp predict_complete_word(word, %{cmu_dict: cmu_dict} = state) do
    case Map.get(cmu_dict, word) do
      nil ->
        # Unknown word: fallback to ML model
        predict_word_with_ml(word, state)

      syllables ->
        # Known word: use dictionary (100% accurate)
        syllables
    end
  end

  defp predict_partial_word(word, state) do
    # For partial words, always use fast ML inference
    predict_word_with_ml(word, state)
  end

  defp predict_word_with_ml(word, %{model: nil}) do
    # No ML model available, use simple heuristic
    simple_syllable_count(word)
  end

  defp predict_word_with_ml(word, %{model: model, char_to_index: char_to_index}) do
    # Encode word as one-hot tensor
    input_tensor = encode_word(word, char_to_index)

    # Run inference with Ortex
    {output} = Ortex.run(model, {input_tensor})

    # Post-process: round and ensure minimum 1 syllable
    syllables = output
                |> Nx.to_flat_list()
                |> hd()
                |> round()
                |> max(1)

    syllables
  end

  defp encode_word(word, char_to_index) do
    # Convert word to lowercase and truncate/pad to max length
    chars = word
            |> String.downcase()
            |> String.graphemes()
            |> Enum.take(@max_word_length)

    # Pad with empty strings if needed
    padded_chars = chars ++ List.duplicate("", @max_word_length - length(chars))

    # Convert to one-hot encoding
    one_hot_matrix = Enum.map(padded_chars, fn char ->
      if char == "" do
        # Empty character -> all zeros
        List.duplicate(0.0, @alphabet_size)
      else
        # Create one-hot vector
        index = Map.get(char_to_index, char, 0)  # Default to first char if unknown
        List.duplicate(0.0, @alphabet_size)
        |> List.replace_at(index, 1.0)
      end
    end)

    # Convert to Nx tensor with shape [1, 18, 28] (batch_size=1)
    one_hot_matrix
    |> Nx.tensor(type: :f32)
    |> Nx.new_axis(0)  # Add batch dimension
  end

  # Simple fallback heuristic for when ML model is unavailable
  defp simple_syllable_count(word) do
    word
    |> String.downcase()
    |> String.replace(~r/[^aeiouy]/, "")
    |> String.length()
    |> max(1)  # Minimum 1 syllable
  end
end
