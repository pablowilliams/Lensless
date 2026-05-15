# Lensless

Three machine learning demos that run entirely in your browser. No server, no API key, no telemetry. The model weights travel one way, from a public CDN to your browser cache, and from then on the inferences happen locally.

**Live:** https://pablowilliams.github.io/Lensless/

## What is here

Three tabs, each driving a different small model from the [Xenova](https://huggingface.co/Xenova) catalogue through [transformers.js](https://huggingface.co/docs/transformers.js).

1. **Image classification.** MobileNet V2 on ImageNet 1000. About 14 MB. Upload or pick a sample image and the model returns the top five labels with confidence scores.
2. **Object detection.** DETR with a ResNet 50 backbone, COCO 91 classes. About 167 MB. Upload an image and the model returns bounding boxes with labels and confidence scores, drawn as SVG over the image.
3. **Sentiment analysis.** DistilBERT fine tuned on SST 2. About 67 MB. Paste a sentence or paragraph and the model returns a positive or negative verdict with a confidence score.

Each tab carries a visible warning of the first time download size, rendered on page load rather than after the user clicks. After the first use of a tab, the model is cached in IndexedDB and subsequent inferences take a few hundred milliseconds.

## Stack

Vanilla HTML, CSS, and an ES module that imports `@huggingface/transformers` from jsDelivr. No build step. The host page is under 30 KB. The models are quantised ONNX and run in WebAssembly by default, or WebGPU when the browser supports it.

## Accessibility

Built to WCAG 2.2 AA. Real APG tabs pattern with roving tabindex and Left, Right, Home, End keyboard support. Every file input has a visible label, an accept hint, and a maximum size. Model downloads report progress as text in a polite live region, with percentage spoken rather than implied by bar width. Bounding boxes are drawn as SVG with a parallel ordered list of detected items naming the object, the confidence, and the position. The image alt text is updated after inference to include the top classification, so it stops being empty as soon as it can be meaningful. Sentiment results are conveyed in text first, with colour and bars as supporting signal only.

## Run locally

```bash
git clone https://github.com/pablowilliams/Lensless.git
cd Lensless
# Any static server will do, since modern browsers block module imports from file://
npx http-server
```

## Notes

The CDN imports come from jsDelivr. If you fork this, you can vendor the library and host it from the same origin to remove the third party dependency. The privacy guarantee is unaffected either way because no inference data ever leaves the browser.
