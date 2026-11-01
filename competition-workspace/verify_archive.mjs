import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'WORKSPACE_ARCHIVE_MANIFEST.json'), 'utf8'));
const buffer = Buffer.alloc(8 * 1024 * 1024);
let checkedBytes = 0;
let checkedParts = 0;
function hashFile(file, expected, aggregate = null) {
  const hash = crypto.createHash('sha256');
  const fd = fs.openSync(file, 'r');
  let size = 0;
  try {
    for (;;) {
      const count = fs.readSync(fd, buffer, 0, buffer.length, null);
      if (count === 0) break;
      const chunk = buffer.subarray(0, count);
      hash.update(chunk);
      if (aggregate) aggregate.update(chunk);
      size += count;
    }
  } finally {
    fs.closeSync(fd);
  }
  if (size !== expected.bytes || hash.digest('hex') !== expected.sha256) {
    throw new Error('Size or SHA256 mismatch: ' + file);
  }
  checkedBytes += size;
  return size;
}
for (const entry of manifest.files) {
  if (entry.storage === 'file') {
    hashFile(path.join(root, entry.path), entry);
  } else if (entry.storage === 'split') {
    const directory = path.join(root, entry.parts_directory);
    const metadata = JSON.parse(fs.readFileSync(path.join(directory, 'parts-manifest.json'), 'utf8'));
    const aggregate = crypto.createHash('sha256');
    let bytes = 0;
    for (const part of metadata.parts) {
      bytes += hashFile(path.join(directory, part.name), part, aggregate);
      checkedParts++;
    }
    if (bytes !== entry.bytes || aggregate.digest('hex') !== entry.sha256) {
      throw new Error('Reconstructed original mismatch: ' + entry.path);
    }
  } else throw new Error('Unknown storage type: ' + entry.storage);
}
for (const entry of manifest.symlinks) {
  const file = path.join(root, entry.path);
  if (fs.readlinkSync(file) !== entry.archived_target || !fs.existsSync(file)) {
    throw new Error('Invalid symlink: ' + entry.path);
  }
}
console.log(JSON.stringify({verified_source_files: manifest.files.length,
  verified_source_bytes: checkedBytes, verified_parts: checkedParts,
  verified_symlinks: manifest.symlinks.length}, null, 2));
