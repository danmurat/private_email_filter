import { Text } from 'react-native';
import { render } from '@testing-library/react-native';

function Greeting() {
  return <Text>Hello, PrivMail</Text>;
}

describe('example', () => {
  it('renders text', async () => {
    const { getByText } = await render(<Greeting />);
    expect(getByText('Hello, PrivMail')).toBeTruthy();
  });
});
