# bustle
a re-creation and extension of _BUSTLE: Botton-up Program Synthesis Through Learning-Guided Exploration_, an [ICLR 2021 paper](https://openreview.net/forum?id=yHeg4PbFHh) by Odena et al.

## Installation
- `pip install -r requirements.txt`
- `pip install git+https://github.com/huggingface/transformers.git`
- `pip install "outlines @ git+https://github.com/outlines-dev/outlines.git@main"`

## TODOs

### Paper

- [x] [`bustle`](bustle.py) implement basic bottom-up synthesis (Algorithm 1 without blue lines)
  - [x] bug fix: check that `Vt` is correctly wrapped when passed to `propertySignature`
  - [ ] think carefully about error handling and program synthesis: can the program make assumptions about inputs? how can we handle `IF(ISNUMBER(X), VALUE(x), 0)`? this requires a sub-expression failing on the whole, but not on the selected cases. Maybe we shouldn't be discarding errors, but have an error signal.
- [x] test implementation on tiny arithmetic language
- [x] add types to DSL operations and extend tiny arithmetic language with boolean types and `if` operation
- [x] allow multiple DSLs
- [x] support dynamic errors (such as division by zero) in DSL execution
- [x] [`stringdsl`](stringdsl.py) implement string manipulation DSL (Figure 1)
  - [x] implemement remaining TODOs in operation execution
  - [ ] for initialization, extract string constants from I/O examples
  - [x] rectify the DSL to match the benchmarks
  - [x] convert property lists from Appendix C
- [ ] implement ML add-ons (Algorithm 1 including blue lines)
  - [x] scaffold the blue lines
  - [x] settle on shapes for `s_io` and `s_vo`, the interfaces between `propertySignature` and `reweightWithModel`
  - [x] implement `propertySignature`
  - [x] support multiple inputs in `propertySignature`
  - [x] implement `reweightWithModel`
  - [x] consider reweighting for the initial values
  - [x] learn model
  - [x] use trained model
    - [ ] make sure the trained model works as expected with multiple inputs
  - [ ] add a validation loss checker (implement early stopping)
  - [ ] try a big run!
- [x] record some metrics (e.g. the number of subexpressions evaluated) during synthesis to compare approaches
    - [ ] trained ML seems to require _more_ subexpressions evaluated!
- [x] re-add the initial string constants and see whether the test cases pass with the ML
- [x] write parser for string DSL
- [x] write a pretty printer
- [ ] use the benchmarks mentioned in the paper
  - [x] Appendix B
    - [x] add some inputs for the benchmark set 
    - [ ] test bustle on a benchmark program
  - [ ] [SysGus](https://github.com/SyGuS-Org/benchmarks)
- [x] generate data from synthesis search (last paragraph of Section 3.1)
  - [ ] generate interesting inputs
  - [ ] persist the data generation

### Beyond Paper
- [ ] see if the property signatures are discriminating
  - [x] manually find two subexpressions, one in and one out, such that the property signatures are the same (found "s" vs "t")
  - [ ] brute force the current benchmarks to find more meaningful discrimination
- [ ] consider a wake/sleep approach
  - [ ] first, use a trained model with `bustle` in `generate_dataset` instead of the brute-force `bustle` without ML
  - [ ] then, iterate back and forth

### Probe Paper
_Just-in-Time Learning for Bottom-Up Enumerative Synthesis_, Shraddha Barke, Hila Peleg, Nadia Polikarpova. OOPSLA'20. ([PDF](https://cseweb.ucsd.edu/~npolikarpova/publications/oopsla20-probe.pdf))

- [ ] implement Probe
  - [x] implement basic skeleton for cost
  - [x] calculate new cost based on partial solutions
    - [ ] debug why all costs are the same

### Literature review angles
- [ ] find the framing that is most accurate and also gives us idea for further improvements
  - [ ] reinforcement learning: learning a policy for how to explore programs and using that policy exploration to improve our policy exploration further
  - [ ] wake/sleep=
  - [ ] generator/discriminator
  - [ ] curriculum learning

### Best practices
- [ ] have flags during training for the different experiments to generate dataset
  - [ ] use vanilla bustle during training
  - [ ] use learnt model during training
  - [ ] `N` parameter as a flag
- [x] build a (DSL-parameterized) program executor
- [ ] improve the DSL interface so that implementations of DSLs are more concise?
- [ ] use a proper Python testing framework?
- [x] save model in a reproduceable way
