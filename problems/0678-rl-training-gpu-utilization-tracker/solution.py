def track_gpu_utilization(timestamps_ms: list, gpu_util_pct: list, phase_labels: list) -> dict:
    durations = [
        timestamps_ms[i + 1] - timestamps_ms[i]
        for i in range(len(timestamps_ms) - 1)
    ]

    total_time = sum(durations)
    if total_time <= 0:
        raise ValueError("Total time must be positive.")

    avg_util = sum(
        duration * util
        for duration, util in zip(durations, gpu_util_pct)
    ) / total_time

    phase_data = {}

    for i, (duration, util, phase) in enumerate(
        zip(durations, gpu_util_pct, phase_labels)
    ):
        if phase not in phase_data:
            phase_data[phase] = {
                "time": 0.0,
                "weighted_util": 0.0,
                "wasted_gpu_time": 0.0,
                "first_index": i,
            }

        stats = phase_data[phase]
        stats["time"] += duration
        stats["weighted_util"] += duration * util
        stats["wasted_gpu_time"] += duration * (100 - util)

    phase_stats = {}

    for phase in sorted(phase_data):
        stats = phase_data[phase]
        phase_stats[phase] = {
            "total_time_ms": round(stats["time"], 2),
            "avg_util_pct": round(stats["weighted_util"] / stats["time"], 2),
            "time_fraction_pct": round(stats["time"] / total_time * 100, 2),
        }

    bottleneck_phase = max(
        phase_data,
        key=lambda phase: (
            phase_data[phase]["wasted_gpu_time"],
            -phase_data[phase]["first_index"],
        ),
    )

    return {
        "total_time_ms": round(total_time, 2),
        "avg_gpu_util_pct": round(avg_util, 2),
        "phase_stats": phase_stats,
        "bottleneck_phase": bottleneck_phase,
        "gpu_idle_fraction_pct": round(100 - avg_util, 2),
    }
    