import { component$, useSignal, useTask$ } from '@builder.io/qwik';
import { countSyllables as countSyllablesRuleBased } from '~/utils/syllable-counter';
import { countSyllables as countSyllablesDecisionTree } from '~/utils/decision-tree-syllable-counter';
import { countSyllables as countSyllablesSwitch, setSyllableCounterType, SyllableCounterType } from '~/utils/syllable-counter-switch';

export default component$(() => {
  const text = useSignal('');
  const ruleBased = useSignal(0);
  const decisionTree = useSignal(0);
  const switchCounter = useSignal(0);
  const selectedCounter = useSignal<SyllableCounterType>(SyllableCounterType.RULE_BASED);

  useTask$(async ({ track }) => {
    const currentText = track(() => text.value);

    // Count syllables using different methods
    ruleBased.value = countSyllablesRuleBased(currentText);
    decisionTree.value = countSyllablesDecisionTree(currentText);

    // Set the selected counter type
    setSyllableCounterType(selectedCounter.value);
    switchCounter.value = await countSyllablesSwitch(currentText);
  });

  return (
    <div class="container mx-auto p-4">
      <h1 class="text-2xl font-bold mb-4">Syllable Counter Test</h1>

      <div class="mb-4">
        <label class="block mb-2">Enter text:</label>
        <textarea
          class="w-full p-2 border rounded"
          rows={5}
          bind:value={text}
          placeholder="Enter text to count syllables"
        ></textarea>
      </div>

      <div class="mb-4">
        <label class="block mb-2">Select counter type:</label>
        <select
          class="p-2 border rounded"
          onChange$={(e) => {
            selectedCounter.value = e.target.value as SyllableCounterType;
          }}
        >
          <option value={SyllableCounterType.RULE_BASED}>Rule-based</option>
          <option value={SyllableCounterType.DECISION_TREE}>Decision Tree</option>
          <option value={SyllableCounterType.ONNX} disabled>ONNX (Disabled)</option>
          <option value={SyllableCounterType.SIMPLE_ONNX} disabled>Simple ONNX (Disabled)</option>
        </select>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="p-4 border rounded">
          <h2 class="text-xl font-bold mb-2">Rule-based</h2>
          <p class="text-3xl">{ruleBased.value}</p>
        </div>

        <div class="p-4 border rounded">
          <h2 class="text-xl font-bold mb-2">Decision Tree</h2>
          <p class="text-3xl">{decisionTree.value}</p>
        </div>

        <div class="p-4 border rounded">
          <h2 class="text-xl font-bold mb-2">Switch ({selectedCounter.value})</h2>
          <p class="text-3xl">{switchCounter.value}</p>
        </div>
      </div>

      <div class="mt-8">
        <h2 class="text-xl font-bold mb-2">Test Words</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
          {[
            'hello', 'world', 'python', 'syllable', 'counter', 'decision', 'tree',
            'haiku', 'poetry', 'japanese', 'tradition', 'seventeen', 'syllables',
            'five', 'seven', 'nature', 'season', 'moment', 'insight'
          ].map((word) => (
            <button
              key={word}
              class="p-2 border rounded hover:bg-gray-100"
              onClick$={() => text.value = word}
            >
              {word}
            </button>
          ))}
        </div>
      </div>

      <div class="mt-8">
        <h2 class="text-xl font-bold mb-2">Test Haikus</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            "Morning sun rises\nDew drops glisten on green leaves\nA new day begins",
            "Typing on keyboard\nThoughts flow into characters\nTasks become haikus",
            "Mountain silhouette\nShadows dance across the lake\nPeace in solitude",
            "Deadline approaching\nFingers race across the keys\nWork becomes a blur",
            "Empty task list waits\nIdeas form in my mind\nTime to write them down"
          ].map((haiku) => (
            <button
              key={haiku}
              class="p-2 border rounded hover:bg-gray-100 text-left"
              onClick$={() => text.value = haiku}
            >
              <pre>{haiku}</pre>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
});
