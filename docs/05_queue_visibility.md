# Queue Message Visibility

This page describes the **Northstar Engineering Reference**, a fictional but internally consistent
technical platform created for SearchWorkbench evaluation. It specifies what the component
must do, how operators should configure it, and which observations are sufficient to verify it.
The contract is designed to be read alongside other pages in this handbook, not as a vendor API.

## Operational model

A consumer leases a message temporarily and acknowledges it only after durable work completes.

The operating environment matters because batch workers handle messages with unpredictable processing duration. Engineers must distinguish the
steady-state request path from the recovery path. Successful behavior means that callers see
a predictable response and that state transitions can be reconstructed from durable records.
Unexpected behavior should be reproduced with controlled traffic, realistic concurrency,
and timestamped observations rather than inferred from a single dashboard screenshot.

The operational rule is implemented at the boundary where requests are accepted. That boundary
must be explicit in design review: an incoming request is admitted, the component checks
local and remote state, and the resulting change is either committed or rejected with a
machine-readable reason. A healthy response is not proof that all downstream side effects
completed; audits should include the component's persistence and delivery acknowledgments.

For incident reconstruction, preserve request identifiers, deployment revision, monotonic
timestamps, and the configuration version that was active when a decision occurred.
Compare samples before and after mitigation using the same request population.

## Configuration reference

visibility_timeout_seconds defaults to 60 in Queue Message Visibility.

The parameter is applied at the service boundary rather than inferred from client behavior.
Changes are versioned and require a staged rollout. Operators should write down the chosen
value, the prior setting, the affected tenants, and a rollback owner before changing production.
Never assume that a lower numeric setting is inherently safer; assess its effect on throughput,
error handling, queues, and recovery time. The example below is an illustrative Northstar
configuration, not a public cloud provider command.

```yaml
apiVersion: northstar.internal/v1
kind: ComponentPolicy
metadata:
  name: queue_visibility
spec:
  visibility_timeout_seconds: "60"
  telemetry:
    auditDecisions: true
    includeRequestIds: true
```

Configuration is validated at startup, published in diagnostics, and attached to every
release artifact. Invalid values fail closed with a descriptive startup error so a typo
does not silently change platform semantics. Downstream clients must not assume the default
is an implicit service-level guarantee across all environments.

## Runbook and failure signatures

An expired lease can cause duplicate processing while the first worker is still running.

The immediate mitigation is specific: Extend the lease for long jobs and make the consumer operation idempotent.

Use a three-stage incident workflow. First, isolate the failing cohort and find the earliest
credible timestamp. Second, compare counters and structured error events at the component
boundary with independent downstream evidence. Third, exercise the rollback in a staging
environment that approximates peak concurrency. Confirm that the solution restores the
user-visible contract, not merely that the alert has stopped firing.

Keep failed requests in a replay-safe diagnostic queue only where access control and
retention policy allow it. Avoid broad, unbounded retries while the failure is active.
A remediation is complete only after the error signature stays below threshold for an
entire observation window and the service owner records the incident outcome.

## Validation and trade-offs

Long visibility windows reduce duplicates but delay retries after worker crashes.

In a pre-release test, simulate the following scenario: batch workers handle messages with unpredictable processing duration. Record a baseline,
then introduce dependency jitter, abrupt process exits, a partial availability event,
and a clean recovery. The test should assert a concrete state outcome, not just an HTTP
status. Track error rate, percentile latency, throughput, queue depth, and any associated
storage mutations. The metrics have different physical meanings and must not be merged
into a single undocumented reliability score.

Use representative load and at least two independent measurements to avoid a misleading
conclusion. A low-volume demonstration can show basic functionality but cannot establish
reliability at production scale. Any performance claim should state traffic shape,
deployment topology, hardware, index size if applicable, and the duration of the sample.

The design review should explicitly answer which subsystem owns the invariant,
what evidence shows that the invariant held, and how the team will detect regression.
Record any trade-off that shifts work or risk from the application to the operator.
