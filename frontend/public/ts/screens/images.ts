import { el, clear, emptyState, spinnerLabel, labeledField, notify, confirmAction, readValue, writeValue, readString, pickDefined, renderParamFields, statusMessage as setStatus, asError, asParamValueMap, asParamValues, messageOf, type MountOptions, type ParamValue, type Route } from '../ui.js';
import { api } from '../api.js';
import type { AddImageModelRequest, DeleteImageModelResponse, GenerateImageRequest, GenerateImageResponse, ImageFile, ImageFileList, ImageModel, ImageModelList, ImageRecipe, ImageRecipeList, InferenceParam, InferenceParamList } from '../api.js';
import { Row, Stack, Split, Grid, Page, Heading, Card, CardHeader, Hint, InlineHint, LineHint, FieldGroup, FieldLabel, Notice, Button, LinkButton, Input, TextArea, Select, Option, Form } from '../components.js';

function defaultsFor(params: InferenceParam[]): Record<string, ParamValue> {
  const values: Record<string, ParamValue> = {};
  for (const param of params) {
    if (param.default !== null) values[param.name] = param.default;
  }
  return values;
}

function fileMatches(file: ImageFile, categories: string[] | undefined): boolean {
  return !categories || categories.length === 0 || categories.includes(file.category);
}

export function mount(_route: Route, _options: MountOptions = {}) {
  const root = Page({ narrow: true });
  root.appendChild(Heading(1, 'Images'));

  const modelCard = Card();
  const modelSelect = Select({});
  const addModelButton = Button('Add model');
  const deleteModelButton = Button('Delete', { variant: 'danger' });
  const modelStatus = Hint();
  const availability = Stack();
  const modelField = labeledField('Model', modelSelect);
  modelCard.appendChild(CardHeader('Image model', Row(addModelButton, deleteModelButton)));
  modelCard.appendChild(modelField.wrapper);
  modelCard.appendChild(modelStatus);
  modelCard.appendChild(availability);

  const addPanel = el('div', { class: 'inset stack', hidden: true });
  modelCard.appendChild(addPanel);

  const generationCard = Card();
  const promptInput = TextArea({
    rows: 4,
    placeholder: 'Describe the image…',
    required: true,
  });
  const promptField = labeledField('Prompt', promptInput);
  const paramsStatus = Hint();
  const paramsGrid = Grid({});
  const negativeContainer = Stack();
  const paramsSection = Stack(paramsStatus, paramsGrid, negativeContainer);
  const generateButton = Button('Generate', { variant: 'primary', submit: true });
  const generateStatus = Hint();
  const generateForm = Form({ novalidate: true }, promptField.wrapper, paramsSection, Row(generateButton), generateStatus);

  const preview = el('div', { class: 'inset image-preview' });
  const previewPlaceholder = Hint('Generated images are not saved and are lost on refresh.');
  preview.appendChild(previewPlaceholder);
  const download = LinkButton('Download', '#', { download: 'myai-image.png', hidden: true });
  const previewCard = Card(CardHeader('Result'), preview, Row(download), LineHint('Each generation replaces the previous image. Images are never stored by the application.'));

  const stage = Split(generationCard, previewCard);
  generationCard.appendChild(CardHeader('Generate'));
  generationCard.appendChild(generateForm);
  root.appendChild(modelCard);
  root.appendChild(stage);

  interface ImagesState {
    models: ImageModel[];
    selectedId: string;
    params: InferenceParam[];
    paramsReady: boolean;
    paramsToken: number;
    paramsByModel: Record<string, Record<string, ParamValue>>;
    busy: boolean;
    recipes: ImageRecipe[];
    files: ImageFile[];
    recipesLoaded: boolean;
    filesLoaded: boolean;
    recipesError: Error | null;
    filesError: Error | null;
    catalogLoading: boolean;
    addBusy: boolean;
  }

  const state: ImagesState = {
    models: [],
    selectedId: '',
    params: [],
    paramsReady: false,
    paramsToken: 0,
    paramsByModel: asParamValueMap(readValue<Record<string, unknown>>('images.params', {})),
    busy: false,
    recipes: [],
    files: [],
    recipesLoaded: false,
    filesLoaded: false,
    recipesError: null,
    filesError: null,
    catalogLoading: false,
    addBusy: false,
  };

  let destroyed = false;

  function selectedModel(): ImageModel | null {
    return state.models.find((model) => model.id === state.selectedId) || null;
  }

  function persistParams(): void {
    writeValue('images.params', state.paramsByModel);
  }

  function updateGenerateState(): void {
    const model = selectedModel();
    const available = Boolean(model && model.available);
    generateButton.disabled = state.busy || !available || !state.paramsReady || !promptInput.value.trim();
    deleteModelButton.disabled = state.busy || !model;
  }

  function renderAvailability(): void {
    clear(availability);
    const model = selectedModel();
    if (!model) return;
    if (model.available) return;
    availability.appendChild(el('div', { class: 'field-error' },
      'This model is unavailable because required files are missing. Generation is disabled.'));
    const list = el('ul', { class: 'missing-list' });
    for (const item of model.missing) list.appendChild(el('li', {}, item));
    availability.appendChild(list);
  }

  function renderModelOptions(): void {
    clear(modelSelect);
    for (const model of state.models) {
      modelSelect.appendChild(Option(model.id, `${model.name || model.id}${model.available ? '' : ' (unavailable)'}`));
    }
    if (state.models.length === 0) {
      modelSelect.appendChild(Option('', 'No image models configured'));
    }
    modelSelect.value = state.selectedId;
    modelSelect.disabled = state.busy || state.models.length === 0;
  }

  function fillModelSelect(): void {
    if (state.models.length === 0) {
      state.selectedId = '';
      renderModelOptions();
      clear(generationCard);
      generationCard.appendChild(CardHeader('Generate'));
      generationCard.appendChild(emptyState(
        'No image models',
        'Add a model preset to start generating images.',
        Button('Add model', { small: true, onClick: openAddPanel }),
      ));
      clear(preview);
      preview.appendChild(previewPlaceholder);
      updateGenerateState();
      return;
    }
    if (!generationCard.querySelector('form')) {
      generationCard.replaceChildren(CardHeader('Generate'), generateForm);
    }
    renderModelOptions();
    updateGenerateState();
  }

  async function loadModels(options: { selectId?: string } = {}): Promise<void> {
    setStatus(modelStatus, 'Loading image models…', { loading: true });
    renderModelOptions();
    modelSelect.disabled = true;
    try {
      const data = await api.get<ImageModelList>('/api/sd-cpp/models');
      if (destroyed) return;
      state.models = Array.isArray(data.models) ? data.models : [];
      const stored = readString('images.model', '');
      const preferred = options.selectId || stored || data.default || (state.models[0] && state.models[0].id) || '';
      state.selectedId = state.models.some((model) => model.id === preferred)
        ? preferred
        : (state.models[0] ? state.models[0].id : '');
      if (state.selectedId) writeValue('images.model', state.selectedId);
      clear(modelStatus);
      fillModelSelect();
      renderAvailability();
      if (state.selectedId) await loadParams(state.selectedId);
      else {
        state.params = [];
        state.paramsReady = false;
        renderParams();
      }
    } catch (error) {
      if (destroyed) return;
      state.models = [];
      state.selectedId = '';
      modelSelect.replaceChildren(Option('', 'Models unavailable'));
      modelSelect.disabled = true;
      setStatus(modelStatus, messageOf(error) || 'Could not load image models.', {
        error: true,
        retry: () => loadModels(options),
      });
      generateButton.disabled = true;
    }
  }

  function currentValuesFor(modelId: string): Record<string, ParamValue> {
    return asParamValues(state.paramsByModel[modelId]);
  }

  function renderParams(): void {
    clear(paramsGrid);
    clear(negativeContainer);
    if (!state.paramsReady) return;
    const model = selectedModel();
    if (!model) return;
    const values = { ...defaultsFor(state.params), ...currentValuesFor(model.id) };
    const negative = state.params.find((param) => param.name === 'negative_prompt');
    const regular = state.params.filter((param) => param.name !== 'negative_prompt');

    const modelId = model.id;

    function commit(name: string, value: ParamValue | undefined): void {
      const stored = { ...currentValuesFor(modelId) };
      if (value === undefined || value === '') delete stored[name];
      else stored[name] = value;
      state.paramsByModel[modelId] = stored;
      persistParams();
    }

    renderParamFields(paramsGrid, regular, values, commit, {
      idPrefix: `image-${model.id}`,
      prefillDefaults: true,
    });

    if (negative) {
      const id = `image-${model.id}-negative_prompt`;
      const area = TextArea({
        id,
        rows: 2,
        value: values.negative_prompt !== undefined ? String(values.negative_prompt) : '',
        placeholder: 'Things to avoid in the image…',
      });
      area.addEventListener('input', () => commit('negative_prompt', area.value));
      negativeContainer.appendChild(FieldGroup(FieldLabel('negative_prompt', id), area, InlineHint('Leave empty to use the model default.')));
    }

    if (state.params.some((param) => param.name === 'seed')) {
      negativeContainer.appendChild(LineHint('A seed of −1 picks a random seed for each generation.'));
    }
  }

  async function loadParams(modelId: string): Promise<void> {
    const token = ++state.paramsToken;
    state.paramsReady = false;
    state.params = [];
    clear(paramsGrid);
    clear(negativeContainer);
    setStatus(paramsStatus, 'Loading generation parameters…', { loading: true });
    updateGenerateState();
    try {
      const data = await api.get<InferenceParamList>(`/api/sd-cpp/inference-params?model=${encodeURIComponent(modelId)}`);
      if (destroyed || token !== state.paramsToken || state.selectedId !== modelId) return;
      state.params = Array.isArray(data) ? data : [];
      state.paramsReady = true;
      clear(paramsStatus);
      renderParams();
      updateGenerateState();
    } catch (error) {
      if (destroyed || token !== state.paramsToken) return;
      state.paramsReady = false;
      setStatus(paramsStatus, messageOf(error) || 'Could not load generation parameters.', {
        error: true,
        retry: () => loadParams(modelId),
      });
      updateGenerateState();
    }
  }

  modelSelect.addEventListener('change', () => {
    state.selectedId = modelSelect.value;
    writeValue('images.model', state.selectedId);
    renderAvailability();
    if (state.selectedId) loadParams(state.selectedId);
    updateGenerateState();
  });

  promptInput.addEventListener('input', updateGenerateState);

  function setFormBusy(busy: boolean): void {
    state.busy = busy;
    promptInput.disabled = busy;
    modelSelect.disabled = busy || state.models.length === 0;
    addModelButton.disabled = busy;
    generateButton.disabled = busy;
    for (const control of paramsGrid.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>('input,select,textarea')) control.disabled = busy;
    for (const control of negativeContainer.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>('input,select,textarea')) control.disabled = busy;
    if (busy) {
      generateButton.replaceChildren(spinnerLabel('Generating…'));
    } else {
      generateButton.textContent = 'Generate';
    }
    updateGenerateState();
  }

  generateForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (state.busy) return;
    const model = selectedModel();
    if (!model || !model.available) return;
    const prompt = promptInput.value.trim();
    if (!prompt) {
      promptInput.focus();
      return;
    }
    clear(generateStatus);
    setFormBusy(true);
    generateStatus.replaceChildren(spinnerLabel('Generating image…'));
    const values = { ...defaultsFor(state.params), ...currentValuesFor(model.id) };
    try {
      const payload: GenerateImageRequest = {
        model: model.id,
        prompt,
        params: pickDefined(values),
      };
      const data = await api.post<GenerateImageResponse>('/api/images/generate', payload);
      if (destroyed) return;
      if (!data.image) throw new Error('The server returned no image.');
      const source = `data:image/png;base64,${data.image}`;
      clear(preview);
      preview.appendChild(el('img', { src: source, alt: `Generated image for: ${prompt}` }));
      download.href = source;
      download.hidden = false;
      clear(generateStatus);
    } catch (error) {
      if (destroyed) return;
      setStatus(generateStatus, messageOf(error) || 'Image generation failed.', { error: true });
      notify(messageOf(error) || 'Image generation failed.', { level: 'error' });
    } finally {
      if (!destroyed) setFormBusy(false);
    }
  });

  deleteModelButton.addEventListener('click', async () => {
    const model = selectedModel();
    if (!model) return;
    const confirmed = await confirmAction({
      title: 'Delete image model?',
      message: `The preset “${model.name}” will be removed. The model files on disk are not touched.`,
      confirmLabel: 'Delete model',
    });
    if (!confirmed) return;
    deleteModelButton.disabled = true;
    try {
      await api.del<DeleteImageModelResponse>(`/api/sd-cpp/models/${encodeURIComponent(model.id)}`);
      notify('Image model deleted.', { level: 'success' });
      await loadModels();
    } catch (error) {
      deleteModelButton.disabled = false;
      notify(messageOf(error) || 'Could not delete the image model.', { level: 'error' });
    }
  });

  // Add-model panel
  const nameInput = Input({ type: 'text', placeholder: 'My image model', required: true });
  const recipeSelect = Select({});
  const nameField = labeledField('Name', nameInput);
  const recipeField = labeledField('Recipe', recipeSelect, { hint: 'The recipe decides which model files are needed.' });
  const recipeDescription = LineHint();
  const slotsGrid = Grid({});
  const requiredAny = el('div', {});
  const addStatus = Hint();
  const addSave = Button('Save model', { variant: 'primary', submit: true });
  const addCancel = Button('Cancel');
  const addForm = Form({ novalidate: true }, nameField.wrapper, recipeField.wrapper, recipeDescription, slotsGrid, requiredAny, addStatus, Row(addSave, addCancel));
  addPanel.appendChild(CardHeader('Add image model'));
  addPanel.appendChild(addForm);

  function setAddStatus(message: string, options: { error?: boolean; loading?: boolean; retry?: (() => void) | null } = {}): void {
    setStatus(addStatus, message, options);
  }

  function renderRecipeSlots(): void {
    clear(slotsGrid);
    clear(requiredAny);
    recipeDescription.textContent = '';
    const recipe = state.recipes.find((item) => item.id === recipeSelect.value);
    if (!recipe) return;
    recipeDescription.textContent = recipe.description || '';
    for (const slot of recipe.slots) {
      const matching = state.files.filter((file) => fileMatches(file, slot.categories));
      const select = Select({});
      select.appendChild(Option('', matching.length ? 'Select a file' : 'No matching files'));
      for (const file of matching) {
        select.appendChild(Option(file.id, `${file.name}${file.category ? ` · ${file.category}` : ''}`));
      }
      select.disabled = matching.length === 0;
      const label = `${slot.label || slot.id}${slot.required ? ' *' : ''}`;
      const hint = slot.categories.length
        ? `Categories: ${slot.categories.join(', ')}`
        : 'Any file category';
      const field = labeledField(label, select, { hint });
      field.wrapper.dataset.slot = slot.id;
      slotsGrid.appendChild(field.wrapper);
    }
    for (const group of recipe.required_any) {
      if (!Array.isArray(group) || group.length === 0) continue;
      const labels = group.map((slotId) => {
        const slot = recipe.slots.find((item) => item.id === slotId);
        return slot?.label || slotId;
      });
      requiredAny.appendChild(Notice({ level: 'warning' },
        `At least one of: ${labels.join(', ')}`));
    }
  }

  function fillRecipeSelect(): void {
    clear(recipeSelect);
    if (state.recipes.length === 0) {
      recipeSelect.appendChild(Option('', 'No recipes available'));
      recipeSelect.disabled = true;
      return;
    }
    recipeSelect.disabled = false;
    for (const recipe of state.recipes) {
      recipeSelect.appendChild(Option(recipe.id, recipe.name || recipe.id));
    }
    renderRecipeSlots();
  }

  async function loadCatalog(): Promise<void> {
    if (state.catalogLoading) return;
    state.catalogLoading = true;
    setAddStatus('Loading recipes and model files…', { loading: true });
    const [recipesResult, filesResult] = await Promise.allSettled([
      api.get<ImageRecipeList>('/api/sd-cpp/recipes'),
      api.get<ImageFileList>('/api/sd-cpp/files'),
    ]);
    if (destroyed) return;
    state.catalogLoading = false;
    if (recipesResult.status === 'fulfilled') {
      state.recipes = Array.isArray(recipesResult.value.recipes) ? recipesResult.value.recipes : [];
      state.recipesLoaded = true;
      state.recipesError = null;
    } else {
      state.recipesError = asError(recipesResult.reason);
      state.recipesLoaded = false;
    }
    if (filesResult.status === 'fulfilled') {
      state.files = Array.isArray(filesResult.value.files) ? filesResult.value.files : [];
      state.filesLoaded = true;
      state.filesError = null;
    } else {
      state.filesError = asError(filesResult.reason);
      state.filesLoaded = false;
    }
    if (state.recipesError || state.filesError) {
      const message = [
        state.recipesError ? `Recipes: ${messageOf(state.recipesError)}` : '',
        state.filesError ? `Files: ${messageOf(state.filesError)}` : '',
      ].filter(Boolean).join(' ');
      setAddStatus(message || 'Could not load the image model catalog.', {
        error: true,
        retry: loadCatalog,
      });
      fillRecipeSelect();
      return;
    }
    setAddStatus('');
    fillRecipeSelect();
  }

  function openAddPanel(): void {
    addPanel.hidden = false;
    if (!state.recipesLoaded || !state.filesLoaded) loadCatalog();
    else fillRecipeSelect();
    nameInput.focus();
  }

  function closeAddPanel(): void {
    addPanel.hidden = true;
    setAddStatus('');
    nameInput.value = '';
  }

  addModelButton.addEventListener('click', () => {
    if (addPanel.hidden) openAddPanel();
    else closeAddPanel();
  });
  addCancel.addEventListener('click', closeAddPanel);
  recipeSelect.addEventListener('change', renderRecipeSlots);

  addForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (state.addBusy) return;
    const recipe = state.recipes.find((item) => item.id === recipeSelect.value);
    const name = nameInput.value.trim();
    nameField.setError('');
    if (!name) {
      nameField.setError('Name is required.');
      nameInput.focus();
      return;
    }
    if (!recipe) {
      setAddStatus('Select a recipe.', { error: true });
      return;
    }
    const files: Record<string, string> = {};
    const missing: string[] = [];
    for (const slot of recipe.slots) {
      const field = slotsGrid.querySelector(`[data-slot="${CSS.escape(slot.id)}"]`);
      const select = field?.querySelector('select');
      if (select?.value) files[slot.id] = select.value;
      if (slot.required && !select?.value) missing.push(slot.label || slot.id);
    }
    for (const group of recipe.required_any) {
      if (!Array.isArray(group) || group.length === 0) continue;
      if (!group.some((slotId) => files[slotId])) {
        const labels = group.map((slotId) => {
          const slot = recipe.slots.find((item) => item.id === slotId);
          return slot?.label || slotId;
        });
        missing.push(`one of: ${labels.join(', ')}`);
      }
    }
    if (missing.length > 0) {
      setAddStatus(`Missing required files: ${missing.join('; ')}.`, { error: true });
      return;
    }
    state.addBusy = true;
    addSave.disabled = true;
    addSave.replaceChildren(spinnerLabel('Saving…'));
    setAddStatus('');
    try {
      const payload: AddImageModelRequest = { name, recipe: recipe.id, files };
      const saved = await api.post<ImageModel>('/api/sd-cpp/models', payload);
      if (destroyed) return;
      notify('Image model added.', { level: 'success' });
      closeAddPanel();
      await loadModels({ selectId: saved.id });
    } catch (error) {
      if (destroyed) return;
      setAddStatus(messageOf(error) || 'Could not add the image model.', { error: true });
    } finally {
      if (!destroyed) {
        state.addBusy = false;
        addSave.disabled = false;
        addSave.textContent = 'Save model';
      }
    }
  });

  renderModelOptions();
  updateGenerateState();
  loadModels();

  return {
    element: root,
    destroy() {
      destroyed = true;
      state.paramsToken += 1;
    },
  };
}
