"""
Comprehensive tests for image processing example.

Tests verify:
- End-to-end pipeline execution
- Fire detection accuracy
- Different execution modes
- Node functionality
- Error handling
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add examples to path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

# Import test data generator
from generate_test_data import generate_satellite_images

# Import demo components
from image_processing_demo import (
    ImageLoader,
    ImagePreprocessor,
    FireDetector,
    ResultAggregator,
    create_pipeline
)

from noob.node import Node
from noob.runner import SynchronousRunner


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_image_dir():
    """Create temporary directory with test images"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate 10 test images
        generate_satellite_images(tmpdir, n_images=10)

        yield tmpdir


@pytest.fixture
def sample_image_with_fire():
    """Create a sample image with fire pattern"""
    img = Image.new('RGB', (512, 512), color=(40, 120, 40))  # Green base

    # Add fire hotspot
    pixels = img.load()
    for y in range(200, 300):
        for x in range(200, 300):
            pixels[x, y] = (255, 140, 0)  # Orange fire

    return img


@pytest.fixture
def sample_image_without_fire():
    """Create a sample image without fire"""
    img = Image.new('RGB', (512, 512), color=(40, 120, 40))  # Green only
    return img


# ============================================================================
# Node Tests
# ============================================================================

class TestImageLoader:
    """Test ImageLoader node"""

    def test_load_images(self, temp_image_dir):
        """Test loading images from directory"""
        loader = ImageLoader(str(temp_image_dir), batch_size=5)
        loader.init()

        # Should find 10 images
        assert len(loader.all_images) == 10

        # Test processing
        result = loader.process()

        assert "image_paths" in result
        assert "batch_num" in result
        assert "total_images" in result
        assert len(result["image_paths"]) == 5
        assert result["total_images"] == 10

        loader.deinit()

    def test_multiple_batches(self, temp_image_dir):
        """Test processing multiple batches"""
        loader = ImageLoader(str(temp_image_dir), batch_size=3)
        loader.init()

        batch1 = loader.process()
        assert len(batch1["image_paths"]) == 3

        batch2 = loader.process()
        assert len(batch2["image_paths"]) == 3

        batch3 = loader.process()
        assert len(batch3["image_paths"]) == 3

        batch4 = loader.process()
        assert len(batch4["image_paths"]) == 1  # Last partial batch

        loader.deinit()


class TestImagePreprocessor:
    """Test ImagePreprocessor node"""

    def test_preprocess_image(self, temp_image_dir):
        """Test image preprocessing"""
        preprocessor = ImagePreprocessor()
        preprocessor.init()

        # Get an image path
        image_paths = list(temp_image_dir.glob("*.jpg"))
        assert len(image_paths) > 0

        result = preprocessor.process(image_paths[0])

        assert result is not None
        assert "image_id" in result
        assert "image_data" in result
        assert "has_fire_label" in result

        # Check image data shape and type
        img_data = result["image_data"]
        assert isinstance(img_data, np.ndarray)
        assert img_data.shape == (512, 512, 3)
        assert img_data.min() >= 0.0
        assert img_data.max() <= 1.0

        preprocessor.deinit()

    def test_fire_label_detection(self, temp_image_dir):
        """Test ground truth fire label detection from filename"""
        preprocessor = ImagePreprocessor()
        preprocessor.init()

        # Find an image with fire in name
        fire_images = list(temp_image_dir.glob("*_fire.jpg"))
        normal_images = list(temp_image_dir.glob("*_normal.jpg"))

        if fire_images:
            result = preprocessor.process(fire_images[0])
            assert result["has_fire_label"] is True

        if normal_images:
            result = preprocessor.process(normal_images[0])
            assert result["has_fire_label"] is False

        preprocessor.deinit()


class TestFireDetector:
    """Test FireDetector node"""

    def test_detect_fire(self, sample_image_with_fire, tmp_path):
        """Test fire detection on image with fire"""
        detector = FireDetector()
        detector.init()

        # Save image
        img_path = tmp_path / "test_fire.jpg"
        sample_image_with_fire.save(img_path)

        # Load and preprocess
        img = Image.open(img_path).resize((512, 512))
        img_array = np.array(img) / 255.0

        # Detect
        result = detector.process(
            image_data=img_array,
            image_id="test_fire",
            has_fire_label=True
        )

        assert "fire_detected" in result
        assert "confidence" in result
        assert "fire_percentage" in result
        assert "bounding_boxes" in result

        # Should detect fire
        assert result["fire_detected"] is True
        assert result["confidence"] > 0.0

        detector.deinit()

    def test_no_fire(self, sample_image_without_fire, tmp_path):
        """Test fire detection on image without fire"""
        detector = FireDetector()
        detector.init()

        # Save image
        img_path = tmp_path / "test_normal.jpg"
        sample_image_without_fire.save(img_path)

        # Load and preprocess
        img = Image.open(img_path).resize((512, 512))
        img_array = np.array(img) / 255.0

        # Detect
        result = detector.process(
            image_data=img_array,
            image_id="test_normal",
            has_fire_label=False
        )

        # Should not detect fire
        assert result["fire_detected"] is False

        detector.deinit()

    def test_accuracy_tracking(self, sample_image_with_fire, tmp_path):
        """Test accuracy tracking"""
        detector = FireDetector()
        detector.init()

        # Process image with fire
        img_path = tmp_path / "test.jpg"
        sample_image_with_fire.save(img_path)
        img_array = np.array(Image.open(img_path).resize((512, 512))) / 255.0

        result = detector.process(img_array, "test", has_fire_label=True)

        # Should be correct
        assert result["correct"] == (result["fire_detected"] == result["ground_truth"])

        detector.deinit()


class TestResultAggregator:
    """Test ResultAggregator node"""

    def test_aggregate_results(self):
        """Test result aggregation"""
        aggregator = ResultAggregator()
        aggregator.init()

        # Add some results
        result1 = aggregator.process(
            image_id="img1",
            fire_detected=True,
            confidence=0.9,
            ground_truth=True,
            correct=True,
            fire_percentage=5.0,
            bounding_boxes=[],
            processed_at=0.0
        )

        assert result1["total_processed"] == 1
        assert result1["fires_detected"] == 1

        result2 = aggregator.process(
            image_id="img2",
            fire_detected=False,
            confidence=0.1,
            ground_truth=False,
            correct=True,
            fire_percentage=0.1,
            bounding_boxes=[],
            processed_at=0.0
        )

        assert result2["total_processed"] == 2
        assert result2["fires_detected"] == 1
        assert result2["accuracy"] == 100.0  # Both correct

        aggregator.deinit()

    def test_accuracy_calculation(self):
        """Test accuracy calculation"""
        aggregator = ResultAggregator()
        aggregator.init()

        # 3 correct, 1 incorrect
        for i in range(3):
            aggregator.process(
                image_id=f"img{i}",
                fire_detected=True,
                correct=True,
                ground_truth=True,
                confidence=0.9,
                fire_percentage=5.0,
                bounding_boxes=[],
                processed_at=0.0
            )

        aggregator.process(
            image_id="img3",
            fire_detected=True,
            correct=False,
            ground_truth=False,
            confidence=0.5,
            fire_percentage=1.0,
            bounding_boxes=[],
            processed_at=0.0
        )

        result = aggregator.process()
        assert result["accuracy"] == 75.0  # 3/4 = 75%

        aggregator.deinit()


# ============================================================================
# Pipeline Tests
# ============================================================================

class TestPipeline:
    """Test complete pipeline execution"""

    def test_create_pipeline(self, temp_image_dir):
        """Test pipeline creation"""
        tube = create_pipeline(str(temp_image_dir), batch_size=5)

        # Check nodes
        assert "loader" in tube.nodes
        assert "preprocessor" in tube.nodes
        assert "detector" in tube.nodes
        assert "aggregator" in tube.nodes

        # Check edges
        assert len(tube.edges) > 0

    def test_pipeline_execution(self, temp_image_dir):
        """Test end-to-end pipeline execution"""
        tube = create_pipeline(str(temp_image_dir), batch_size=5)

        runner = SynchronousRunner(tube)
        runner.init()

        # Run for 2 epochs
        results = []
        for _ in range(2):
            result = runner.process()
            if result:
                results.append(result)

        runner.deinit()

        # Should have results
        assert len(results) > 0

        # Check final result
        final_result = results[-1]
        assert "total_processed" in final_result
        assert final_result["total_processed"] > 0

    def test_pipeline_processes_all_images(self, temp_image_dir):
        """Test that pipeline processes all images"""
        n_images = len(list(temp_image_dir.glob("*.jpg")))

        tube = create_pipeline(str(temp_image_dir), batch_size=3)

        runner = SynchronousRunner(tube)
        runner.init()

        # Run enough epochs to process all images
        n_epochs = (n_images // 3) + 2
        final_result = None

        for _ in range(n_epochs):
            result = runner.process()
            if result:
                final_result = result

        runner.deinit()

        # All images should be processed
        assert final_result is not None
        assert final_result["total_processed"] == n_images


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests"""

    def test_fire_detection_accuracy(self, temp_image_dir):
        """Test overall fire detection accuracy"""
        tube = create_pipeline(str(temp_image_dir), batch_size=10)

        runner = SynchronousRunner(tube)
        runner.init()

        # Run pipeline
        results = []
        for _ in range(2):  # Process all images
            result = runner.process()
            if result:
                results.append(result)

        runner.deinit()

        # Check accuracy
        final_result = results[-1]
        assert "accuracy" in final_result

        # Should have reasonable accuracy (>50% at least)
        assert final_result["accuracy"] > 50.0

    def test_performance_metrics(self, temp_image_dir):
        """Test performance tracking"""
        tube = create_pipeline(str(temp_image_dir), batch_size=5)

        runner = SynchronousRunner(tube)

        import time
        start_time = time.time()

        runner.init()

        for _ in range(2):
            runner.process()

        runner.deinit()

        elapsed = time.time() - start_time

        # Should complete reasonably fast
        assert elapsed < 30.0  # 30 seconds max for 10 images


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling"""

    def test_empty_directory(self, tmp_path):
        """Test handling of empty directory"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = ImageLoader(str(empty_dir), batch_size=5)
        loader.init()

        assert len(loader.all_images) == 0

        result = loader.process()
        assert len(result["image_paths"]) == 0

        loader.deinit()

    def test_invalid_image(self, tmp_path):
        """Test handling of invalid image file"""
        # Create invalid image file
        invalid_file = tmp_path / "invalid.jpg"
        invalid_file.write_text("not an image")

        preprocessor = ImagePreprocessor()
        preprocessor.init()

        result = preprocessor.process(invalid_file)

        # Should return None for invalid images
        assert result is None

        preprocessor.deinit()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
