# eVED A2 failure diagnosis and enhanced feature design

## Confirmed A2 failure

The completed A2 run did not learn a useful input-output relation:

- A2 validation RMSE: 11.7884 kW.
- Training-mean constant predictor RMSE on the same validation split: 11.7896 kW.
- A2 correlation: 0.0051.
- The implemented covariance image had mean absolute magnitude 3.75e-6.
- Covariance-channel standard deviations for tractive effort and elevation were
  approximately 1.11e-6 and 1.45e-6.

The root implementation error was the extra division of the 100x100 outer
product by 99. It is not specified by the paper's image construction and made
the input almost zero after the already-normalized series was centered. The
CNN therefore converged to the training-target mean. Covariance centering also
removes absolute operating levels by construction, so constant speed, steady
tractive effort, constant auxiliary load, OAT, and SOC cannot be distinguished
from zero using the image alone.

The earlier A2 result is therefore an invalid reproduction result and must not
be presented as evidence that the paper architecture fails on eVED.

As a read-only signal check, a ridge model was fitted to 20,000 sampled
training windows using only the enhanced operating-level features and evaluated
on 10,000 validation windows. It obtained 8.92 kW RMSE and 0.635 correlation,
versus 11.55 kW for the constant-mean predictor on that sample. This is not a
final experiment result, but it verifies that the added target-independent
levels contain real predictive signal that the old covariance images discarded.

## Remaining domain differences after the implementation fix

- The paper used a single simulated Nissan Leaf for training and dynamometer
  data for testing. eVED contains real-world, multi-vehicle data.
- The current tractive-effort reconstruction uses one Nissan Leaf parameter
  set for every eVED vehicle; unknown mass, drag, rolling resistance, and
  drivetrain efficiency create irreducible model error.
- The paper randomly split 10-second partitions. This project uses a harder,
  leakage-free trip-disjoint split.
- Near-zero-net-energy trips make percentage energy deviation numerically
  unstable; RMSE, MAE, correlation, and absolute trip-energy error must be
  reported alongside it.

## Mechanistic enhanced features

The battery-power balance is approximated by wheel mechanical power adjusted
for propulsion/regen efficiency plus auxiliary power. This motivates:

1. Speed and acceleration: inertial, rolling, and aerodynamic demand.
2. Road gradient and elevation: gravitational potential change.
3. Jerk: transient driving and rapid torque changes not represented by mean
   acceleration alone.
4. Tractive effort: combined inertia, rolling resistance, drag, and grade.
5. Positive wheel power `max(F_trac * v, 0)`: propulsion demand.
6. Regenerative wheel power `max(-F_trac * v, 0)`: separated because regen and
   propulsion efficiencies are asymmetric.
7. Auxiliary power: direct non-traction battery load.
8. Outside-air temperature: HVAC demand and temperature-dependent efficiency.
9. Initial SOC: voltage and regenerative-power acceptance operating state.

No enhanced feature uses battery current, voltage, target power, target-window
statistics, or future observations. Scaling is fitted on training trips only.

For A1, nine time-varying features use the original multi-branch extraction
geometry, while mean OAT and initial SOC enter as constants. For A2, the nine
time-varying series form covariance images and an 11-value level branch carries
their means plus mean OAT and initial SOC. This branch restores the DC operating
information that covariance centering necessarily removes.
