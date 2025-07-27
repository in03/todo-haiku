defmodule Mix.Tasks.BigPhoney do
  use Mix.Task

  @shortdoc "Test big-phoney microservice integration"

  @moduledoc """
  This task provides commands to test the big-phoney microservice integration.

  ## Examples

      mix big_phoney test          # Test the microservice connection
      mix big_phoney health        # Check microservice health
      mix big_phoney count "hello" # Count syllables in a word
      mix big_phoney haiku "text"  # Test haiku endpoint

  """

  def run(args) do
    Application.ensure_all_started(:todo_haiku)

    case args do
      ["test"] ->
        test_microservice()

      ["health"] ->
        check_health()

      ["count", word] ->
        count_syllables(word)

      ["haiku", text] ->
        test_haiku_endpoint(text)

      _ ->
        IO.puts("""
        Unknown command. Available commands:
          mix big_phoney test          # Test the microservice connection
          mix big_phoney health        # Check microservice health
          mix big_phoney count "hello" # Count syllables in a word
          mix big_phoney haiku "text"  # Test haiku endpoint
        """)
    end
  end

  defp test_microservice do
    IO.puts("🧪 Testing Big Phoney Microservice...")
    IO.puts("=" <> String.duplicate("=", 49))

    # Test basic functionality
    case TodoHaiku.BigPhoneyClient.test() do
      {:ok, syllables} ->
        IO.puts("✅ Microservice is working correctly!")
        IO.puts("   Endpoint: #{TodoHaiku.BigPhoneyClient.endpoint()}")
        IO.puts("   Test word 'hello' has #{syllables} syllables")

      {:error, reason} ->
        IO.puts("❌ Microservice test failed:")
        IO.puts("   Error: #{inspect(reason)}")
        IO.puts("")
        IO.puts("💡 Make sure the microservice is deployed and running.")
        IO.puts("   Run: cd big-phoney-api && ./deploy.sh")
    end
  end

  defp check_health do
    IO.puts("🏥 Checking Big Phoney Microservice Health...")

    case TodoHaiku.BigPhoneyClient.health_check() do
      {:ok, status} ->
        IO.puts("✅ Microservice is healthy!")
        IO.puts("   Status: #{inspect(status)}")

      {:error, reason} ->
        IO.puts("❌ Microservice health check failed:")
        IO.puts("   Error: #{inspect(reason)}")
    end
  end

  defp count_syllables(word) do
    IO.puts("📝 Counting syllables for: '#{word}'")

    case TodoHaiku.BigPhoneyClient.count_syllables(word) do
      {:ok, syllables} ->
        IO.puts("✅ Result: #{syllables} syllables")

      {:error, reason} ->
        IO.puts("❌ Failed: #{inspect(reason)}")
    end
  end

  defp test_haiku_endpoint(text) do
    IO.puts("📝 Testing haiku endpoint with: '#{text}'")

    case TodoHaiku.BigPhoneyClient.count_syllables_haiku(text) do
      {:ok, result} ->
        IO.puts("✅ Haiku endpoint result:")
        IO.puts("   Text: #{result["text"]}")
        IO.puts("   Lines:")
        Enum.each(result["lines"], fn line ->
          IO.puts("     - '#{line["line"]}': #{line["syllables"]} syllables")
        end)

      {:error, reason} ->
        IO.puts("❌ Failed: #{inspect(reason)}")
    end
  end
end
