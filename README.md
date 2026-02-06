Working Homomorphic Encryption demo.

Files are executable .py files meant to "simulate" real deployment. So we
have some email, encrypt it, "send it to server" (which has the HE ML model),
run a prediction, "send it back to the client", we decrypt and see an accurate
classification (Spam or not Spam).

Of course this is all on the same device and there's no HTTP requests being made.
But it can be done if needed (though this is excessive for us for the time being).
