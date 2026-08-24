import { Image } from 'expo-image';
import { Link } from 'expo-router';
import { Linking, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

export default function About() {
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.container}>
      <Link href="/" style={[styles.backLink, { top: insets.top + 12 }]}>
        ← Back
      </Link>

      <ScrollView
        contentContainerStyle={[styles.content, { paddingTop: insets.top + 72 }]}
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.title}>About PrivMail</Text>

        <Text style={styles.body}>
          This is a demo of ‘homomorphic encryption’ classifiying incoming encrypted
          emails. The server is still able to figure out, despite receiving a jumbled up
          ciphertext, if the email is spam or not. In a real application, your email would either be filtered out 
          or sent to you correctly, all without being able to see what was in those emails.
        </Text>

        <Text style={styles.heading}>Why?</Text>

        <Text style={styles.body}>
          Emails you recieve can be sensitive; and perhaps you’d prefer many to not be seen if
          someone decided to take a look, for whatever reasons. Much of this privacy concern is solved when
          you surf the web, thanks to https (no one can snoop and see what data is being sent and
          recieved). This is great! But consider our spam filtering use case:
        </Text>

        <Text style={styles.body}>
          You encrypt your email.{`\n`}
          You send it.{`\n`}
          The server recieves.{`\n`}
          The server decrypts…
        </Text>

        <Text style={styles.body}>
          To filter spam emails, we use certain AI models to figure out which are good or bad. But
          the AI models can’t read encrypted text! They wouldn’t be able to figure out what is good
          or bad, because it’s all jumbled nonsense! Hence the need to decrypt, which defeats many of
          the privacy and security purposes of encrypting.
        </Text>

        <Text style={styles.body}>
          This is true for ANYTHING that uses AI. Spam filtering, health analytics, chatgpt and the
          like. Whatever you send MUST be decrypted; it MUST be plaintext, otherwise they wouldn’t
          work.
        </Text>

        <Text style={styles.body}>
          Currently there’s an enormous amount of data being held by any company that offers any
          type of AI service, that is in plaintext, or decryptable, which is a huge risk.
        </Text>

        <Text style={styles.body}>
          But if you use homomorphic encryption, this problem is gone. Your data remains encrypted, and
          they can still offer whatever service they provide. This will be increasingly important as
          AI services continue to grow.
        </Text>

        <Text style={styles.heading}>How?</Text>

        <Text style={styles.body}>
          As simply as I can put it, consider the below image:
        </Text>

        <Image
          source={require('../assets/images/grouphomomorphism.png')}
          style={styles.diagram}
          contentFit="contain"
          accessibilityLabel="Group homomorphism diagram"
        />

        <Text style={styles.body}>
          Maths has this thing called a ‘group homomorphism’, which is a property about a function.
          You have a function that ‘preserves’ some algebraic structure (say addition, or
          multiplication, or both, etc..). If you do something like a * b, then apply the result to
          f, and it is equal to doing f(a) * f(b) (applying f before to each), then the function is
          homomorphic.
        </Text>

        <Text style={styles.body}>
          This is the magical ‘thing’ that happens. If you make f an encryption function (jumble all
          the words), while keeping it homomorphic with certain operations, you’d be able to compute
          things correctly! You’d be able to run an AI model on it! because it will be still able to
          make a correct result (because it is homomorphic). When you decrypt (which only you can
          do), you can see the true result calculated.
        </Text>

        <Text style={styles.heading}>Further notes</Text>

        <Text style={styles.body}>
          This project was initially my dissertation for university{' '}
          <Text
            style={styles.inlineLink}
            onPress={() => Linking.openURL('https://zenodo.org/records/21030838')}
            accessibilityRole="link"
          >
            https://zenodo.org/records/21030838
          </Text>{' '}
          exploring what can be applied with the current Homomorphic encryption libraries and how to
          incorporate them into our Machine Learning models. I decided to try and ‘deploy’ this
          project onto the internet. I figured it would be useful to learn the skills to do so, and
          that this is the type of thing that would eventually happen if you are a business wanting
          to make use of something like this, so it’d be worth ‘proving’ that this can be deployed in
          the real world. Of course, spam filtering is exactly the thing you’d bring to a website;
          but the point is to show that it can work (a more suitible application could be related to
          something financial, where you can make use of homomorphic encryption with your customers
          data that they may input on your website). Regardless, we have a server that runs the model
          on an AWS instance, which recieves the encrypted data sent over http from the website; and it works. It
          does not handle a large scale of course (i just wanted to quickly showcase), but doing so
          wouldn’t be difficult.
        </Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f2f2f7',
    paddingHorizontal: 20,
  },
  backLink: {
    position: 'absolute',
    left: 20,
    zIndex: 1,
    fontSize: 14,
    fontWeight: '600',
    color: '#007AFF',
  },
  content: {
    alignItems: 'center',
    paddingBottom: 40,
  },
  title: {
    fontSize: 30,
    fontWeight: '700',
    color: '#1c1c1e',
    letterSpacing: -0.5,
    textAlign: 'center',
    marginBottom: 24,
  },
  heading: {
    width: '100%',
    maxWidth: 700,
    alignSelf: 'center',
    marginTop: 28,
    marginBottom: 10,
    fontSize: 22,
    fontWeight: '700',
    color: '#1c1c1e',
    textAlign: 'left',
  },
  body: {
    width: '100%',
    maxWidth: 700,
    alignSelf: 'center',
    marginTop: 14,
    fontSize: 16,
    lineHeight: 25,
    color: '#636366',
    textAlign: 'left',
  },
  inlineLink: {
    color: '#007AFF',
    textDecorationLine: 'underline',
  },
  diagram: {
    width: '100%',
    maxWidth: 700,
    aspectRatio: 1746 / 965,
    marginTop: 18,
  },
});
