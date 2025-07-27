defmodule TodoHaiku.BigPhoneyClient do
  @moduledoc """
  Client for the big-phoney Python microservice.
  Provides syllable counting functionality using the big-phoney ML library.
  """

  @endpoint Application.compile_env(:todo_haiku, :big_phoney_endpoint, "https://your-app-name.fly.dev")

  @doc """
  Count syllables in text using the big-phoney microservice.

  Returns {:ok, syllable_count} on success or {:error, reason} on failure.
  """
  def count_syllables(text) when is_binary(text) do
        case Req.post("#{@endpoint}/syllables/simple",
                  json: %{text: text},
                  receive_timeout: 5000) do
      {:ok, %{status: 200, body: %{"syllables" => syllables}}} ->
        {:ok, syllables}

      {:ok, %{status: status, body: body}} ->
        {:error, "API returned status #{status}: #{inspect(body)}"}

      {:error, reason} ->
        {:error, "Request failed: #{inspect(reason)}"}
    end
  end

  @doc """
  Count syllables in text with detailed word breakdown.

  Returns {:ok, %{syllables: total, words: word_list}} on success or {:error, reason} on failure.
  """
  def count_syllables_detailed(text) when is_binary(text) do
        case Req.post("#{@endpoint}/syllables",
                  json: %{text: text},
                  receive_timeout: 5000) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, %{
          syllables: body["syllables"],
          words: body["words"]
        }}

      {:ok, %{status: status, body: body}} ->
        {:error, "API returned status #{status}: #{inspect(body)}"}

      {:error, reason} ->
        {:error, "Request failed: #{inspect(reason)}"}
    end
  end

  @doc """
  Check if the big-phoney microservice is healthy.

  Returns {:ok, status} on success or {:error, reason} on failure.
  """
  def health_check do
    case Req.get("#{@endpoint}/health", receive_timeout: 3000) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, body}

      {:ok, %{status: status}} ->
        {:error, "Health check failed with status #{status}"}

      {:error, reason} ->
        {:error, "Health check request failed: #{inspect(reason)}"}
    end
  end

  @doc """
  Count syllables for multiple words efficiently.

  Takes a list of words and returns a map of word -> syllable count.
  """
  def count_syllables_batch(words) when is_list(words) do
    # Join words with spaces for batch processing
    text = Enum.join(words, " ")

    case count_syllables_detailed(text) do
      {:ok, %{words: word_data}} ->
        # Convert the word data to a map
        word_data
        |> Enum.map(fn %{"word" => word, "syllables" => syllables} -> {word, syllables} end)
        |> Map.new()

      {:error, _reason} ->
        # Return empty map if microservice fails
        Map.new()
    end
  end

  @doc """
  Count syllables in haiku text with line-by-line breakdown.

  Returns {:ok, %{lines: line_data}} on success or {:error, reason} on failure.
  """
  def count_syllables_haiku(text) when is_binary(text) do
    case Req.post("#{@endpoint}/syllables/haiku",
                  json: %{text: text},
                  receive_timeout: 10000) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, body}

      {:ok, %{status: status, body: body}} ->
        {:error, "API returned status #{status}: #{inspect(body)}"}

      {:error, reason} ->
        {:error, "Request failed: #{inspect(reason)}"}
    end
  end

  @doc """
  Count syllables in haiku text asynchronously.

  Returns a Task reference that can be awaited.
  """
  def count_syllables_haiku_async(text) when is_binary(text) do
    Task.async(fn -> count_syllables_haiku(text) end)
  end

  @doc """
  Get the microservice endpoint URL.
  """
  def endpoint, do: @endpoint

  @doc """
  Test the microservice with a simple word.
  """
  def test do
    test_word = "hello"
    IO.puts("Testing big-phoney microservice with word: '#{test_word}'")

    case count_syllables(test_word) do
      {:ok, syllables} ->
        IO.puts("✅ Success: '#{test_word}' has #{syllables} syllables")
        {:ok, syllables}

      {:error, reason} ->
        IO.puts("❌ Error: #{inspect(reason)}")
        {:error, reason}
    end
  end
end
