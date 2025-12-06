from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/app/logs/scheduler.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_performance_monitoring():
    """Run weekly performance monitoring"""
    logger.info("=" * 60)
    logger.info("Starting performance monitoring")
    logger.info("=" * 60)

    try:
        from src.monitoring.performance_tracking import main

        main()
        logger.info("✓ Performance monitoring completed successfully")
    except Exception as e:
        logger.error(f"❌ Performance monitoring failed: {e}", exc_info=True)


def run_drift_detection():
    """Run weekly drift detection"""
    logger.info("=" * 60)
    logger.info("Starting drift detection")
    logger.info("=" * 60)

    try:
        from src.monitoring.drift_detection import main

        main()
        logger.info("✓ Drift detection completed successfully")
    except Exception as e:
        logger.error(f"❌ Drift detection failed: {e}", exc_info=True)


def health_check():
    """Periodic health check"""
    logger.info(f"Health check at {datetime.now()}")


def main():
    """Setup and start scheduler"""

    logger.info("=" * 60)
    logger.info("FRAUD DETECTION MONITORING SCHEDULER")
    logger.info("=" * 60)
    logger.info("Starting scheduler in Docker container...")

    scheduler = BlockingScheduler()

    # Performance monitoring - Every Monday at 2 AM
    scheduler.add_job(
        run_performance_monitoring,
        trigger=CronTrigger(day_of_week="mon", hour=2, minute=0),
        id="performance_monitoring",
        name="Weekly Performance Monitoring",
        replace_existing=True,
    )
    logger.info("✓ Scheduled: Performance monitoring (Monday 2 AM)")

    # Drift detection - Every Monday at 3 AM
    scheduler.add_job(
        run_drift_detection,
        trigger=CronTrigger(day_of_week="mon", hour=3, minute=0),
        id="drift_detection",
        name="Weekly Drift Detection",
        replace_existing=True,
    )
    logger.info("✓ Scheduled: Drift detection (Monday 3 AM)")

    # Health check - Every hour
    scheduler.add_job(
        health_check, trigger=CronTrigger(minute=0), id="health_check", name="Hourly Health Check", replace_existing=True
    )
    logger.info("✓ Scheduled: Health check (every hour)")

    # For testing: Run monitoring immediately on startup (optional)
    # Uncomment these to test right away
    # logger.info("\n🧪 Running initial monitoring tests...")
    # run_performance_monitoring()
    # run_drift_detection()

    logger.info("\n" + "=" * 60)
    logger.info("SCHEDULER READY")
    logger.info("=" * 60)
    scheduler.print_jobs()
    logger.info("\nPress Ctrl+C to stop")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
