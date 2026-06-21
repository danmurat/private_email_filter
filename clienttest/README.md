For testing our server.

The client code will be a part of our frontend when we get to it. That will be the
process of making a request, sending the required data, and getting an a prediction privately.

### todo

(the test.jsonl is 6mb, so why is our 'reduced' preprocessed test data 40mb? First thought is
probably saved with many zeroes 'not sparse'. But our non-reduced is ~20mb, which has 3000+ 
dimensions. Find out what's going on there at some point, because 40mb is a bit bloated).
