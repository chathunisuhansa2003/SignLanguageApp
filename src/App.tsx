import { useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import * as tf from '@tensorflow/tfjs';

const CAPTURE_MS = 2000;
const MODEL_FRAMES = 16;
const FEATURES_PER_FRAME = 128;
const MODEL_URL = 'indexeddb://sign-language-pilot';
const PILOT_SIGNERS = new Set(['S01', 'S03', 'S04']);

type SampleLabel = 'thank_you' | 'no_sign';
type Point = { x: number; y: number; z: number };
type Frame = {
  elapsedMs: number;
  hands: { handedness: string; points: Point[] }[];
};
type Sample = { signerId: string; label: SampleLabel; frames: Frame[] };
type Capture = {
  mode: 'training' | 'test';
  startedAt: number;
  sample: Sample;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isSample(value: unknown): value is Sample {
  return (
    isRecord(value) &&
    typeof value.signerId === 'string' &&
    value.signerId.trim().length > 0 &&
    (value.label === 'thank_you' || value.label === 'no_sign') &&
    Array.isArray(value.frames) &&
    value.frames.length >= 2 &&
    value.frames.every(
      (frame: unknown) =>
        isRecord(frame) &&
        isFiniteNumber(frame.elapsedMs) &&
        frame.elapsedMs >= 0 &&
        Array.isArray(frame.hands) &&
        frame.hands.length <= 2 &&
        frame.hands.every(
          (hand: unknown) =>
            isRecord(hand) &&
            typeof hand.handedness === 'string' &&
            Array.isArray(hand.points) &&
            hand.points.length === 21 &&
            hand.points.every(
              (point: unknown) =>
                isRecord(point) &&
                isFiniteNumber(point.x) &&
                isFiniteNumber(point.y) &&
                isFiniteNumber(point.z),
            ),
        ),
    )
  );
}

function mergeSamples(samples: Sample[]): Sample[] {
  const unique = new Map<string, Sample>();

  for (const sample of samples) {
    const key = JSON.stringify(
      sample.frames.map(({ elapsedMs, hands }) => [
        elapsedMs,
        hands.map(({ handedness, points }) => [
          handedness,
          points.map(({ x, y, z }) => [x, y, z]),
        ]),
      ]),
    );

    if (!unique.has(key)) unique.set(key, sample);
  }

  return [...unique.values()];
}

function sampleToSequence(sample: Sample): number[][] {
  return Array.from({ length: MODEL_FRAMES }, (_, index) => {
    const frameIndex = Math.round(
      (index * (sample.frames.length - 1)) / (MODEL_FRAMES - 1),
    );

    return ['left', 'right'].flatMap((side) => {
      const hand = sample.frames[frameIndex].hands.find(
        (item) => item.handedness.toLowerCase() === side,
      );

      return hand
        ? [1, ...hand.points.flatMap(({ x, y, z }) => [x, y, z])]
        : Array<number>(64).fill(0);
    });
  });
}

async function predictSample(sample: Sample): Promise<number> {
  await tf.ready();
  const model = await tf.loadLayersModel(MODEL_URL);

  try {
    return tf.tidy(() => {
      const input = tf.tensor3d([sampleToSequence(sample)]);
      const output = model.predict(input) as tf.Tensor;
      return output.dataSync()[0];
    });
  } finally {
    model.dispose();
  }
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cameraRequestRef = useRef(0);
  const captureRef = useRef<Capture | null>(null);

  const [camera, setCamera] = useState<'off' | 'starting' | 'on'>('off');
  const [tracking, setTracking] = useState<'off' | 'loading' | 'ready' | 'failed'>('off');
  const [cameraError, setCameraError] = useState('');

  const [signerId, setSignerId] = useState('S01');
  const [label, setLabel] = useState<SampleLabel>('thank_you');
  const [samples, setSamples] = useState<Sample[]>([]);
  const [captureMode, setCaptureMode] = useState<Capture['mode'] | null>(null);
  const [importing, setImporting] = useState(false);
  const [training, setTraining] = useState(false);
  const [sampleMessage, setSampleMessage] = useState('');
  const [modelMessage, setModelMessage] = useState('');

  const busy = captureMode !== null || importing || training;
  const canCapture = camera === 'on' && tracking === 'ready' && !busy;
  const thankYouCount = samples.filter((sample) => sample.label === 'thank_you').length;

  useEffect(() => {
    return () => {
      cameraRequestRef.current += 1;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (camera !== 'on') return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!video || !canvas || !context) return;

    const preview = video;
    const overlay = canvas;
    const drawing = context;
    let cancelled = false;
    let animationId = 0;
    let lastVideoTime = -1;
    let detector: HandLandmarker | undefined;

    async function finishCapture(capture: Capture) {
      const { sample, mode } = capture;

      try {
        if (sample.frames.length < 2) {
          setSampleMessage('Too few camera frames. Try again.');
          return;
        }

        if (mode === 'test') {
          const score = await predictSample(sample);
          if (!cancelled) {
            const prediction = score >= 0.5 ? 'Thank you' : 'No sign';
            setSampleMessage(
              `Pilot prediction: ${prediction} · Thank you score: ${score.toFixed(2)}`,
            );
          }
          return;
        }

        const hasHand = sample.frames.some((frame) => frame.hands.length > 0);
        if (sample.label === 'thank_you' && !hasHand) {
          setSampleMessage('No hand detected. Try again.');
          return;
        }

        setSamples((current) => mergeSamples([...current, sample]));
        setSampleMessage(`Sample added. ${sample.frames.length} frames captured.`);
      } catch (cause) {
        console.error(cause);
        if (!cancelled) {
          setSampleMessage(
            'Could not run the saved model. Use the same browser and address used for training.',
          );
        }
      } finally {
        if (!cancelled) setCaptureMode(null);
      }
    }

    function drawFrame(timestamp: number) {
      if (cancelled || !detector) return;

      try {
        if (
          preview.readyState >= 2 &&
          preview.videoWidth > 0 &&
          preview.currentTime !== lastVideoTime
        ) {
          lastVideoTime = preview.currentTime;

          if (overlay.width !== preview.videoWidth || overlay.height !== preview.videoHeight) {
            overlay.width = preview.videoWidth;
            overlay.height = preview.videoHeight;
          }

          const result = detector.detectForVideo(preview, timestamp);
          drawing.clearRect(0, 0, overlay.width, overlay.height);
          drawing.fillStyle = '#22e58b';

          for (const hand of result.landmarks) {
            for (const point of hand) {
              drawing.beginPath();
              drawing.arc(point.x * overlay.width, point.y * overlay.height, 4, 0, Math.PI * 2);
              drawing.fill();
            }
          }

          const capture = captureRef.current;
          if (capture) {
            const elapsedMs = timestamp - capture.startedAt;

            if (elapsedMs >= 0 && elapsedMs <= CAPTURE_MS) {
              capture.sample.frames.push({
                elapsedMs: Math.round(elapsedMs),
                hands: result.landmarks.map((hand, index) => ({
                  handedness: result.handedness[index]?.[0]?.categoryName ?? 'Unknown',
                  points: hand.map(({ x, y, z }) => ({ x, y, z })),
                })),
              });
            }
          }
        }

        const capture = captureRef.current;
        if (capture && timestamp - capture.startedAt >= CAPTURE_MS) {
          captureRef.current = null;
          void finishCapture(capture);
        }

        animationId = requestAnimationFrame(drawFrame);
      } catch (cause) {
        console.error(cause);
        captureRef.current = null;
        setCaptureMode(null);
        setSampleMessage('');
        setTracking('failed');
        setCameraError('Hand tracking stopped. Restart the camera to try again.');
        drawing.clearRect(0, 0, overlay.width, overlay.height);
      }
    }

    async function startTracking() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm',
        );
        if (cancelled) return;

        const loaded = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          },
          runningMode: 'VIDEO',
          numHands: 2,
        });

        if (cancelled) {
          loaded.close();
          return;
        }

        detector = loaded;
        setTracking('ready');
        animationId = requestAnimationFrame(drawFrame);
      } catch (cause) {
        if (cancelled) return;
        console.error(cause);
        setTracking('failed');
        setCameraError('Could not load hand tracking. Check your connection and restart the camera.');
      }
    }

    void startTracking();

    return () => {
      cancelled = true;
      cancelAnimationFrame(animationId);
      detector?.close();
      drawing.clearRect(0, 0, overlay.width, overlay.height);
    };
  }, [camera]);

  async function startCamera() {
    const requestId = ++cameraRequestRef.current;
    setCameraError('');
    setCamera('starting');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      const video = videoRef.current;
      if (requestId !== cameraRequestRef.current || !video) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;
      video.srcObject = stream;
      setTracking('loading');
      setCamera('on');
    } catch (cause) {
      if (requestId !== cameraRequestRef.current) return;
      console.error(cause);
      setCamera('off');
      setCameraError('Could not open the camera. Check browser permissions.');
    }
  }

  function stopCamera() {
    captureRef.current = null;
    setCaptureMode(null);
    setSampleMessage('');
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) videoRef.current.srcObject = null;

    setTracking('off');
    setCamera('off');
    setCameraError('');
  }

  function startCapture(mode: Capture['mode']) {
    const code = signerId.trim().toUpperCase();
    if (!canCapture || captureRef.current || (mode === 'training' && !code)) return;

    captureRef.current = {
      mode,
      startedAt: performance.now(),
      sample: { signerId: code, label, frames: [] },
    };

    setCaptureMode(mode);

    if (mode === 'test') {
      setSampleMessage('Testing… perform the sign or move normally.');
    } else {
      setSampleMessage(
        label === 'thank_you'
          ? 'Capturing… perform the sign now.'
          : 'Capturing… stay still or move normally.',
      );
    }
  }

  function downloadSamples() {
    const blob = new Blob([JSON.stringify(samples, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');

    link.href = url;
    link.download = 'sign-samples.json';
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function importSamples(files: File[]) {
    setImporting(true);
    setSampleMessage('Importing samples…');

    try {
      const imported: Sample[] = [];

      for (const file of files) {
        const data: unknown = JSON.parse(await file.text());

        if (!Array.isArray(data) || !data.every(isSample)) {
          throw new Error(`${file.name} contains invalid samples.`);
        }

        for (const sample of data) {
          const code = sample.signerId.trim().toUpperCase();
          if (PILOT_SIGNERS.has(code)) {
            imported.push({ ...sample, signerId: code });
          }
        }
      }

      setSamples((current) => mergeSamples([...current, ...imported]));
      setSampleMessage(
        imported.length > 0
          ? 'Import complete. Check the sample counts below.'
          : 'No samples found for S01, S03, or S04.',
      );
    } catch (cause) {
      setSampleMessage(cause instanceof Error ? cause.message : 'Could not import samples.');
    } finally {
      setImporting(false);
    }
  }

  async function trainModel() {
    if (busy) return;

    const trainingSamples = samples.filter((sample) => PILOT_SIGNERS.has(sample.signerId));
    if (
      !trainingSamples.some((sample) => sample.label === 'thank_you') ||
      !trainingSamples.some((sample) => sample.label === 'no_sign')
    ) {
      setModelMessage('Add both labels for the pilot signers S01, S03, and S04.');
      return;
    }

    setTraining(true);
    setModelMessage('Training…');

    let model: tf.Sequential | undefined;
    let inputs: tf.Tensor3D | undefined;
    let labels: tf.Tensor2D | undefined;

    try {
      await tf.ready();
      model = tf.sequential();

      model.add(
        tf.layers.conv1d({
          inputShape: [MODEL_FRAMES, FEATURES_PER_FRAME],
          filters: 8,
          kernelSize: 3,
          activation: 'relu',
        }),
      );
      model.add(tf.layers.flatten());
      model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
      model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });

      inputs = tf.tensor3d(trainingSamples.map(sampleToSequence));
      labels = tf.tensor2d(
        trainingSamples.map((sample) => [sample.label === 'thank_you' ? 1 : 0]),
      );

      await model.fit(inputs, labels, {
        epochs: 20,
        batchSize: Math.min(4, trainingSamples.length),
        shuffle: true,
      });

      await model.save(MODEL_URL);
      setModelMessage(`Model trained on ${trainingSamples.length} samples and saved in this browser.`);
    } catch (cause) {
      setModelMessage(cause instanceof Error ? cause.message : 'Training failed.');
    } finally {
      inputs?.dispose();
      labels?.dispose();
      model?.dispose();
      setTraining(false);
    }
  }
    async function exportModel() {
    let model: tf.LayersModel | undefined;

    try {
      model = await tf.loadLayersModel(MODEL_URL);
      await model.save('downloads://sign-language-pilot');
      setModelMessage('Model exported. Check your browser downloads.');
    } catch (error) {
      setModelMessage(
        error instanceof Error ? error.message : 'Could not export the saved model.',
      );
    } finally {
      model?.dispose();
    }
  }

  return (
    <main style={{ width: '100%', maxWidth: 720, margin: '40px auto', padding: 16, boxSizing: 'border-box' }}>
      <h1>Sign Language Camera</h1>

      <div style={{ position: 'relative', background: '#151922' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          aria-label="Local camera preview"
          style={{
            display: 'block',
            width: '100%',
            height: 'auto',
            aspectRatio: camera === 'on' ? 'auto' : '16 / 9',
          }}
        />
        <canvas
          ref={canvasRef}
          aria-hidden="true"
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
          }}
        />
      </div>

      <p role="status">
        {camera === 'starting' ? 'Requesting camera access…' : `Camera is ${camera}.`}
      </p>
      <p role="status">
        {{
          off: 'Hand tracking is off.',
          loading: 'Loading hand tracking…',
          ready: 'Hand tracking is ready. Show your hands to the camera.',
          failed: 'Hand tracking is unavailable.',
        }[tracking]}
      </p>
      {cameraError && <p role="alert">{cameraError}</p>}

      <button onClick={startCamera} disabled={camera !== 'off'}>Start Camera</button>{' '}
      <button onClick={stopCamera} disabled={camera !== 'on'}>Stop Camera</button>

      <section>
        <h2>Training Samples</h2>

        <p>
          <label>
            Signer code{' '}
            <input
              value={signerId}
              onChange={(event) => setSignerId(event.target.value)}
              disabled={busy}
            />
          </label>
        </p>

        <p>
          <label>
            Sample label{' '}
            <select
              value={label}
              onChange={(event) => setLabel(event.target.value as SampleLabel)}
              disabled={busy}
            >
              <option value="thank_you">Thank you</option>
              <option value="no_sign">No sign</option>
            </select>
          </label>
        </p>

        <button
          onClick={() => startCapture('training')}
          disabled={!canCapture || !signerId.trim()}
        >
          {captureMode === 'training' ? 'Capturing…' : 'Capture Sample'}
        </button>{' '}
        <button onClick={downloadSamples} disabled={samples.length === 0 || busy}>
          Download Samples
        </button>

        <p>
          <label>
            Import Samples{' '}
            <input
              type="file"
              accept=".json,application/json"
              multiple
              disabled={busy}
              onChange={(event) => {
                const files = Array.from(event.currentTarget.files ?? []);
                event.currentTarget.value = '';
                if (files.length > 0) void importSamples(files);
              }}
            />
          </label>
        </p>

        <p>Samples collected: {samples.length}</p>
        <p>Thank you: {thankYouCount} · No sign: {samples.length - thankYouCount}</p>

        <button onClick={() => void trainModel()} disabled={busy || samples.length === 0}>
          {training ? 'Training…' : 'Train Model'}
        </button>{' '}
                <button type="button" onClick={exportModel} disabled={busy}>
          Export Model
        </button>
        <button onClick={() => startCapture('test')} disabled={!canCapture}>
          {captureMode === 'test' ? 'Testing…' : 'Test Live'}
        </button>

        <p role="status">{modelMessage}</p>
        <p role="status">{sampleMessage}</p>
      </section>
    </main>
  );
}