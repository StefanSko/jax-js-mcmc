# jax-js Memory Management Guide

## Overview

jax-js uses **move semantics** for arrays. Every operation **consumes** its input arrays. This is fundamentally different from typical JavaScript where objects are passed by reference and persist.

## The Core Rule

**Every operation consumes its inputs.** If you need to use an array more than once, you must explicitly take a reference with `.ref`.

```typescript
const x = np.array([1, 2, 3]);

// WRONG - x is consumed by first mul, second use fails
const bad = x.mul(2).add(x);  // Error: array freed

// CORRECT - take ref before first use
const good = x.ref.mul(2).add(x);  // Works
```

## How `.ref` Works

- `.ref` increments the reference count and returns the same array object
- Each operation decrements the ref count when it consumes an input
- When ref count reaches 0, the array is freed
- Arrays start with ref count of 1

```typescript
const x = np.array([1, 2, 3]);  // refCount = 1

const y = x.ref;                 // refCount = 2, y === x
const z = y.mul(2);              // y consumed, refCount = 1

// x is still valid (refCount = 1)
const w = x.mul(3);              // x consumed, refCount = 0
// x is now freed
```

## Common Patterns

### Pattern 1: Using an array multiple times in one expression

```typescript
// WRONG
const result = x.mul(x);  // x consumed twice

// CORRECT
const result = x.ref.mul(x);  // ref survives first mul
```

### Pattern 2: Function that will be called multiple times

```typescript
// Caller must pass .ref if they want to keep their array
function processArray(arr: Array): Array {
  return arr.mul(2);  // consumes arr
}

const x = np.array([1, 2, 3]);
const r1 = processArray(x.ref);  // pass ref, x survives
const r2 = processArray(x.ref);  // pass ref again
const r3 = processArray(x);      // final use, no ref needed
```

### Pattern 3: Loop that updates a variable

```typescript
let x = np.array([1, 2, 3]);

for (let i = 0; i < 3; i++) {
  // x.ref keeps x valid for assignment
  x = x.ref.mul(2);
}
// x is now [8, 16, 24]
```

### Pattern 4: tree.map consumes tree leaves

```typescript
import { tree } from '@jax-js/jax';

let params = { w: np.array([1, 2]), b: np.array([3]) };

// tree.map consumes all leaves
params = tree.map((x) => x.mul(2), params);

// To keep original, use tree.ref
const original = { w: np.array([1, 2]), b: np.array([3]) };
const doubled = tree.map((x) => x.mul(2), tree.ref(original));
// original is still valid
```

### Pattern 5: grad() and traced functions

```typescript
const f = (x: Array) => x.ref.mul(x).sum();  // ref needed inside!
const df = grad(f);

const x = np.array([1, 2, 3]);
const gradient = df(x);  // x is consumed
// x is now freed - cannot use again
```

## Gotchas

### 1. Binary operations consume BOTH operands

```typescript
// WRONG
const z = x.add(y);  // both x and y are consumed

// If you need x or y after:
const z = x.ref.add(y);  // x survives
// or
const z = x.add(y.ref);  // y survives
// or
const z = x.ref.add(y.ref);  // both survive
```

### 2. Method chaining is fine (each step produces new array)

```typescript
// This is fine - each operation produces a new array
const result = x.mul(2).add(3).sum();  // x consumed once at start
```

### 3. Returning consumed arrays

```typescript
// WRONG - q is consumed by gradLogProb, can't return it
function broken(q) {
  const grad = gradLogProb(q);  // consumes q
  return [q, grad];  // q is freed!
}

// CORRECT - use ref if you need to return and use
function fixed(q) {
  const grad = gradLogProb(tree.ref(q));  // ref consumed, q survives
  return [q, grad];
}
```

### 4. Arrays in closures

```typescript
// WRONG - x captured but may be freed
const x = np.array([1, 2, 3]);
const fn = () => x.mul(2);  // x might be freed by the time fn is called

// CORRECT - take ref when capturing
const x = np.array([1, 2, 3]);
const xRef = x.ref;
const fn = () => xRef.mul(2);
```

## Debugging Tips

1. **Error message**: `"Referenced tracer Array:float32[N] freed, please use .ref move semantics"`
   - An array was used after it was consumed
   - Add `.ref` before the operation that consumes it

2. **Check ref count**: `x.refCount` shows current count (works even on freed arrays)

3. **Trace consumption**: Each operation consumes inputs. Track which operations run first.

## Summary Table

| Scenario | Solution |
|----------|----------|
| Use array twice in expression | `x.ref.op(x)` |
| Pass to function, keep using | `fn(x.ref)` |
| Update in loop | `x = x.ref.op(...)` |
| Return array after using | `gradFn(tree.ref(q)); return q` |
| Keep original after tree.map | `tree.map(fn, tree.ref(original))` |
| Capture in closure | Take ref before closure creation |
