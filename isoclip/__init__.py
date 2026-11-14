"""IsoCLIP projections and explicitly composed descriptor adaptation."""
from .adaptation import DescriptorTransform, fit_transform
from .projection import image_projection

__all__ = ['DescriptorTransform', 'fit_transform', 'image_projection']
