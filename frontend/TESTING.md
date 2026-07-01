# Testing

This project uses [Jest](https://jestjs.io/) via the [`jest-expo`](https://docs.expo.dev/develop/unit-testing/)
preset, plus [`@testing-library/react-native`](https://callstack.github.io/react-native-testing-library/) for
rendering and querying components.

## Running tests

```sh
npm test
```

This runs Jest in watch mode (re-runs affected tests as you edit files) — the intended day-to-day TDD loop.

Other useful variants:

```sh
# run once and exit (e.g. for CI)
npx jest --ci

# run once with a coverage report
npx jest --coverage

# run only tests matching a name/path pattern
npx jest Login
```

## Where tests live

Tests live in a top-level `__tests__/` directory (not inside `app/`). Expo Router treats every
`.ts`/`.tsx` file under `app/` as a route, so test files placed there would get picked up and
rendered as screens. Jest's default `testMatch` already picks up anything under `__tests__/`, or
any `*.test.ts`/`*.test.tsx` file elsewhere, with no extra config needed.

For a new feature, either:
- add a test file under `__tests__/`, mirroring the source path (e.g. `__tests__/components/Button.test.tsx`), or
- co-locate it next to the source as `Button.test.tsx`.

## Writing a test

`@testing-library/react-native` v14's `render()` is **async** — always `await` it:

```tsx
import { Text } from 'react-native';
import { render } from '@testing-library/react-native';

function Greeting() {
  return <Text>Hello, PrivMail</Text>;
}

describe('Greeting', () => {
  it('renders the greeting text', async () => {
    const { getByText } = await render(<Greeting />);
    expect(getByText('Hello, PrivMail')).toBeTruthy();
  });
});
```

See `__tests__/example.test.tsx` for the working example above.

## Notes for this project specifically

- Components that use `react-native-webview` or `react-native-safe-area-context` (e.g. the SEAL
  encryption WebView bridge) will likely need mocks/providers wrapped around them in tests — the
  example test above deliberately avoids native modules to keep the base setup simple. Add mocks
  as those components get tests.
- No `babel.config.js` is checked in on purpose — `jest-expo` auto-resolves Expo's own Babel preset
  when one isn't present. If you add a `babel.config.js` for another reason, make sure it still
  applies `babel-preset-expo`.
