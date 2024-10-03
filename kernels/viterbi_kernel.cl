// viterbi_kernel.cl
__kernel void viterbi_step(__global const float* prevProbs,
                           __global const float* transProbs,
                           __global const float* emitProbs,
                           __global float* newProbs,
                           const unsigned int numStates,
                           const unsigned int numObservations) {
    int state = get_global_id(0);
    int obs = get_global_id(1);

    if (state >= numStates || obs >= numObservations) return;

    float maxProb = -INFINITY;
    for (int prevState = 0; prevState < numStates; prevState++) {
        float prob = prevProbs[prevState] +
                     transProbs[prevState * numStates + state] +
                     emitProbs[state * numObservations + obs];
        maxProb = max(maxProb, prob);
    }

    newProbs[state * numObservations + obs] = maxProb;
}