const form = document.getElementById("predictionForm");
const predictButton = document.getElementById("predictButton");
const sampleButton = document.getElementById("sampleButton");
const loadingIndicator = document.getElementById("loadingIndicator");
const errorMessage = document.getElementById("errorMessage");
const qualityBadge = document.getElementById("qualityBadge");
const qualityScore = document.getElementById("qualityScore");
const resultCaption = document.getElementById("resultCaption");

const sliderInputs = document.querySelectorAll(".feature-slider");
const numberInputs = document.querySelectorAll(".feature-number");

function formatValue(value) {
    const numericValue = Number(value);
    if (Number.isNaN(numericValue)) {
        return value;
    }
    return numericValue.toFixed(1).replace(/\.0$/, "");
}

function syncDisplay(name, value) {
    const display = document.getElementById(`${name}_value`);
    if (display) {
        display.textContent = formatValue(value);
    }
}

function syncPairedInput(targetName, value, sourceClass) {
    const selector = sourceClass === "feature-slider" ? ".feature-number" : ".feature-slider";
    const pairedInput = document.querySelector(`${selector}[data-target="${targetName}"]`);
    if (pairedInput) {
        pairedInput.value = value;
    }
    syncDisplay(targetName, value);
}

function attachSync(inputs, className) {
    inputs.forEach((input) => {
        syncDisplay(input.dataset.target, input.value);
        input.addEventListener("input", (event) => {
            syncPairedInput(event.target.dataset.target, event.target.value, className);
        });
    });
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}

function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}

function setLoading(isLoading) {
    loadingIndicator.classList.toggle("hidden", !isLoading);
    predictButton.disabled = isLoading;
    predictButton.textContent = isLoading ? "Predicting..." : "Predict Quality";
}

function setResult(label, score) {
    const normalized = label.toLowerCase();
    qualityBadge.textContent = label;
    qualityBadge.className = `quality-badge ${normalized}`;
    qualityScore.textContent = score;
    resultCaption.textContent = `The notebook-based Random Forest model predicts this wine belongs to the ${label} quality band.`;
}

function collectPayload() {
    const payload = {};

    for (const input of numberInputs) {
        const { target } = input.dataset;
        const min = Number(input.min);
        const max = Number(input.max);
        const value = Number(input.value);

        if (Number.isNaN(value)) {
            throw new Error(`Please enter a valid number for ${target.replaceAll("_", " ")}.`);
        }

        if (value < min || value > max) {
            throw new Error(`Keep ${target.replaceAll("_", " ")} between ${formatValue(min)} and ${formatValue(max)}.`);
        }

        payload[target] = value;
    }

    return payload;
}

async function predictQuality() {
    clearError();
    setLoading(true);

    try {
        const payload = collectPayload();
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        setResult(data.quality_label, data.quality_score);
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

function applySampleValues() {
    numberInputs.forEach((numberInput) => {
        const { target, default: defaultValue } = numberInput.dataset;
        const sliderInput = document.querySelector(`.feature-slider[data-target="${target}"]`);
        numberInput.value = defaultValue;
        if (sliderInput) {
            sliderInput.value = defaultValue;
        }
        syncDisplay(target, defaultValue);
    });

    clearError();
}

function drawFeatureImportanceChart() {
    const canvas = document.getElementById("importanceChart");
    const context = canvas.getContext("2d");
    const values = window.featureImportance || [];

    context.clearRect(0, 0, canvas.width, canvas.height);

    const padding = { top: 20, right: 20, bottom: 28, left: 120 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const barHeight = 16;
    const gap = 10;
    const maxValue = Math.max(...values.map((item) => item.importance), 0.01);

    context.font = "12px Manrope";
    values.forEach((item, index) => {
        const y = padding.top + index * (barHeight + gap);
        const width = (item.importance / maxValue) * chartWidth;

        context.fillStyle = "rgba(255,255,255,0.18)";
        context.fillRect(padding.left, y, chartWidth, barHeight);

        const gradient = context.createLinearGradient(padding.left, y, padding.left + width, y);
        gradient.addColorStop(0, "#f3d59b");
        gradient.addColorStop(1, "#c58b37");
        context.fillStyle = gradient;
        context.fillRect(padding.left, y, width, barHeight);

        context.fillStyle = "#f4efe6";
        context.textAlign = "right";
        context.fillText(item.feature.replaceAll("_", " "), padding.left - 12, y + 12);

        context.textAlign = "left";
        context.fillText(item.importance.toFixed(3), padding.left + width + 8, y + 12);
    });
}

attachSync(sliderInputs, "feature-slider");
attachSync(numberInputs, "feature-number");
drawFeatureImportanceChart();

predictButton.addEventListener("click", predictQuality);
sampleButton.addEventListener("click", applySampleValues);
form.addEventListener("submit", (event) => {
    event.preventDefault();
    predictQuality();
});
