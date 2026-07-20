# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from typing import Any

from context_foundry.fusion.schemas import InternalDetection

logger = logging.getLogger(__name__)

class ProtocolValidator(ABC):
    """
    Abstract Base Class for all protocol-specific gatekeepers.
    Enforces the 'Validate then Normalize' pipeline.
    """

    @abstractmethod
    def validate(self, raw_payload: dict[str, Any]) -> tuple[bool, str]:
        """
        Checks if the incoming dictionary strictly matches the protocol's schema.

        Returns:
            Tuple[bool, str]: (Is_Valid, Error_Message_If_Any)
        """
        pass

    @abstractmethod
    def normalize(self, raw_payload: dict[str, Any]) -> InternalDetection:
        """
        Extracts data from the protocol-specific payload and converts it
        into the universal InternalDetection format.

        Returns:
            InternalDetection: The clean, flattened Pydantic model.
        """
        pass

    def process_message(self, raw_payload: dict[str, Any]) -> InternalDetection | None:
        """
        Standardized ingestion pipeline. Call this method from your stream reader.
        """
        is_valid, error_msg = self.validate(raw_payload)

        if not is_valid:
            logger.warning(f"Message rejected by {self.__class__.__name__}: {error_msg}")
            return None

        try:
            return self.normalize(raw_payload)
        except Exception as e:
            logger.error(f"Normalization failed in {self.__class__.__name__}: {e!s}")
            return None
