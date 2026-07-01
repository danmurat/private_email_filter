### Encrypted inference demo

The front end will be an interface for users to use the project. They can select between the
preset test set emails, or paste their own one in. Clientside we will preprocess, encrypt and
send POST request to server, receive result and indicate to user the classification.

### Cross platform? (react native with expo)

Unlikely for now. Things needed client side:
- preprocessing
- encryption

Will need to find a way to use sklearn's TFIDF preprocessor on the client text; and be able to encrypt
decrypt with SEALs library functions, requiring c++.

SEAL already has a library i think that allows you to use it's functions through webassembly in react
(so we can encrypt/decrypt). (node-seal library).

TFIDF needs only to transform (we already fit it). This turns out to be trivial, because all we need
is to pass the 'weight's of the model and transform, which doesn't require python (simpler, and no
need to include stuff like pyodide for python on web). Just need to confirm transforming on web
and locally gives equivalent vectors. This can be done on web and mobile.

Mobile with SEAL is trickier. React native didn't support native wasm, but there's a latest sdk that
does! though will take more work and testing (unstable). Easiest method right now is using WebView
(set it up, send data to the webview 'still local', and do encryption/decryption through that wasm).

Think we're going to have to reduce dimensions of our vectorised email too (defaults to 3020). 
SVD reduce already fit with python; just need to transform, so should be trivial in typescript
like tfidf.

## sum-up:
- set up react native expo project in this directory
- design our front end with relevant elements
- handle transformation of data
- handle node-seal on web wasm and webview on mobile
- make sure design maps to mobile and web dimensions
- handle http request to interact with server


## some things to note:

"node-seal is Emscripten-compiled, so every wasm-backed object (Plaintext, Ciphertext, Context, keys, 
Encryptor/Decryptor/Evaluator, Encoder, etc.) holds heap memory that JS garbage collection never touches
— you must call .delete() yourself or it leaks"
-- when dealing with encryption/decryption stuff, always delete memory!!

"Use the single-threaded node-seal build. The multithreaded variant needs SharedArrayBuffer,
which requires cross-origin isolation (COOP/COEP headers) — you won't get that in a WebView 
loading a local HTML asset. Single-threaded avoids the whole problem."
-- not entirely sure what this exactly means or why this is the case, but consider before importing anything.

WebView needs to be kept mounted/warm, because there's a cold start to this! (mobile only).

The bridge to web view uses 'portMessage' i think. Potentially chunk text when sending over through this
(if too big), and measure how much time this takes (hopefully not a bottleneck).

Diverging of mobile/web. Mobile needs webview for the wasm stuff. Web does not. The web view code must
only apply when targetting mobile.


### starting and running the project

Expo Go is simplest and should work with what we're trying to do. It includes webview as a native module; 
and allows us to run/test the app on our phone without installing any simulators, going on developer plans, etc.

npx create-expo-app@latest .
npx expo install react-native-webview

npx expo start
(this should have a qr, scan it, be on the same network as mac, and app should open).
npx expo start --web
(for web version; or just press w on the first start)
