import type { HistoryEntry } from "./types";

const ITERATIONS = 250_000;
const FILE_MAGIC = new Uint8Array([0x53, 0x52, 0x47, 0x48, 0x01]);

export type VaultEnvelope = {
  format: "srg-history";
  version: 1;
  algorithm: "AES-GCM";
  iterations: number;
  salt: string;
  iv: string;
  ciphertext: string;
};

function toBase64(bytes: Uint8Array) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function fromBase64(value: string) {
  const binary = atob(value);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function asArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

async function deriveKey(password: string, salt: Uint8Array, iterations: number) {
  const source = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(password),
    "PBKDF2",
    false,
    ["deriveKey"],
  );
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", hash: "SHA-256", salt: asArrayBuffer(salt), iterations },
    source,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
}

export async function encryptHistory(entries: HistoryEntry[], password: string): Promise<VaultEnvelope> {
  if (password.length < 8) throw new Error("Use at least 8 characters for the history password.");
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(password, salt, ITERATIONS);
  const plaintext = new TextEncoder().encode(JSON.stringify({ entries }));
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv: asArrayBuffer(iv) },
    key,
    asArrayBuffer(plaintext),
  );
  return {
    format: "srg-history",
    version: 1,
    algorithm: "AES-GCM",
    iterations: ITERATIONS,
    salt: toBase64(salt),
    iv: toBase64(iv),
    ciphertext: toBase64(new Uint8Array(ciphertext)),
  };
}

export async function decryptHistory(envelope: VaultEnvelope, password: string): Promise<HistoryEntry[]> {
  if (envelope.format !== "srg-history" || envelope.version !== 1 || envelope.algorithm !== "AES-GCM") {
    throw new Error("This file is not a supported SRG-Tracker history vault.");
  }
  try {
    const key = await deriveKey(password, fromBase64(envelope.salt), envelope.iterations);
    const plaintext = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: asArrayBuffer(fromBase64(envelope.iv)) },
      key,
      asArrayBuffer(fromBase64(envelope.ciphertext)),
    );
    const payload = JSON.parse(new TextDecoder().decode(plaintext)) as { entries?: HistoryEntry[] };
    return Array.isArray(payload.entries) ? payload.entries : [];
  } catch {
    throw new Error("The password is incorrect or the history file is damaged.");
  }
}

export function parseEnvelopeFile(buffer: ArrayBuffer): VaultEnvelope {
  const bytes = new Uint8Array(buffer);
  if (bytes.length <= FILE_MAGIC.length || !FILE_MAGIC.every((value, index) => bytes[index] === value)) {
    throw new Error("This is not a supported binary .srg-history file.");
  }
  return JSON.parse(new TextDecoder().decode(bytes.slice(FILE_MAGIC.length))) as VaultEnvelope;
}

export function serializeEnvelope(envelope: VaultEnvelope): ArrayBuffer {
  const payload = new TextEncoder().encode(JSON.stringify(envelope));
  const output = new Uint8Array(FILE_MAGIC.length + payload.length);
  output.set(FILE_MAGIC);
  output.set(payload, FILE_MAGIC.length);
  return asArrayBuffer(output);
}

export async function readHistoryFile(): Promise<VaultEnvelope | null> {
  const response = await fetch("/api/history");
  if (response.status === 404) return null;
  if (!response.ok) throw new Error("The local history vault could not be read.");
  return parseEnvelopeFile(await response.arrayBuffer());
}

export async function saveHistoryFile(envelope: VaultEnvelope) {
  const response = await fetch("/api/history", {
    method: "PUT",
    headers: { "Content-Type": "application/octet-stream" },
    body: serializeEnvelope(envelope),
  });
  if (!response.ok) throw new Error("The local history vault could not be saved.");
}

export function downloadEnvelope(envelope: VaultEnvelope) {
  const blob = new Blob([serializeEnvelope(envelope)], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `srg-history-${new Date().toISOString().slice(0, 10)}.srg-history`;
  link.click();
  URL.revokeObjectURL(url);
}
