# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Modificado para añadir soporte de backbone DINOv2 sin registros (dinov2_vit_*)
# Basado en: src/anomalib/models/image/dinomaly/torch_model.py
# Referencia original: guojiajeremy/Dinomaly (CVPR 2025, MIT License)

"""PyTorch model for the Dinomaly model implementation.

Based on PyTorch Implementation of "Dinomaly" by guojiajeremy
Reference: guojiajeremy/Dinomaly  License: MIT

See also:
    :class:`anomalib.models.image.dinomaly.lightning_model.Dinomaly`:
        Dinomaly Lightning model.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

if TYPE_CHECKING:
    from anomalib.data.dataclasses.torch import InferenceBatch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ENCODER_CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
# Registro centralizado de todos los backbones ViT soportados.
#
# Claves soportadas:
#   CON registros (comportamiento original):
#     dinov2reg_vit_small_14, dinov2reg_vit_base_14 (default),
#     dinov2reg_vit_large_14, dinov2reg_vit_giant_14
#
#   SIN registros (NUEVOS en este parche):
#     dinov2_vit_small_14, dinov2_vit_base_14,
#     dinov2_vit_large_14, dinov2_vit_giant_14
# ─────────────────────────────────────────────────────────────────────────────
ENCODER_CONFIGS: dict[str, dict] = {
    # ── DINOv2 CON registros ─────────────────────────────────────────────────
    "dinov2reg_vit_small_14": {
        "hub_repo": "facebookresearch/dinov2",
        "hub_name": "dinov2_vits14_reg",
        "embed_dim": 384,
        "num_heads": 6,
        "target_layers_default": [2, 3, 4, 5, 6, 7, 8, 9],
        "target_layers_large": None,
    },
    "dinov2reg_vit_base_14": {
        "hub_repo": "facebookresearch/dinov2",
        "hub_name": "dinov2_vitb14_reg",
        "embed_dim": 768,
        "num_heads": 12,
        "target_layers_default": [2, 3, 4, 5, 6, 7, 8, 9],
        "target_layers_large": None,
    },
    "dinov2reg_vit_large_14": {
        "hub_repo": "facebookresearch/dinov2",
        "hub_name": "dinov2_vitl14_reg",
        "embed_dim": 1024,
        "num_heads": 16,
        "target_layers_default": [4, 6, 8, 10, 12, 14, 16, 18],
        "target_layers_large": [4, 6, 8, 10, 12, 14, 16, 18],
    },
    "dinov2reg_vit_giant_14": {
        "hub_repo": "facebookresearch/dinov2",
        "hub_name": "dinov2_vitg14_reg",
        "embed_dim": 1536,
        "num_heads": 24,
        "target_layers_default": [4, 6, 8, 10, 12, 14, 16, 18],
        "target_layers_large": [4, 6, 8, 10, 12, 14, 16, 18],
    },

    # ── DINOv2 SIN registros (NUEVOS) ────────────────────────────────────────
    "dinov2_vit_small_14": {
        "hub_repo": "facebookresearch/dinov2",
        "hub_name": "dinov2_vits14",
        "embed_dim": 384,
        "num_heads": 6,
        "target_layers_default": [2, 3, 4, 5, 6, 7, 8, 9],
        "target_layers_large": None,
    },
    "dinov2_vit_base_14": {
        "hub_repo": "facebookresearch/dinov2",
        "hub_name": "dinov2_vitb14",
        "embed_dim": 768,
        "num_heads": 12,
        "target_layers_default": [2, 3, 4, 5, 6, 7, 8, 9],
        "target_layers_large": None,
    },
    "dinov2_vit_large_14": {
        "hub_repo": "facebookresearch/dinov2",
        "hub_name": "dinov2_vitl14",
        "embed_dim": 1024,
        "num_heads": 16,
        "target_layers_default": [4, 6, 8, 10, 12, 14, 16, 18],
        "target_layers_large": [4, 6, 8, 10, 12, 14, 16, 18],
    },
    "dinov2_vit_giant_14": {
        "hub_repo": "facebookresearch/dinov2",
        "hub_name": "dinov2_vitg14",
        "embed_dim": 1536,
        "num_heads": 24,
        "target_layers_default": [4, 6, 8, 10, 12, 14, 16, 18],
        "target_layers_large": [4, 6, 8, 10, 12, 14, 16, 18],
    },
    
    # ── DINOv3 SIN registros (NUEVOS) ────────────────────────────────────────
    "dinov3_vit_base_16": {
        "hub_repo": "facebookresearch/dinov3",
        "hub_name": "dinov3_vitb16",
        "embed_dim": 768,
        "num_heads": 12,
        "target_layers_default": [2, 3, 4, 5, 6, 7, 8, 9],
        "target_layers_large": None,
    }
}


def _load_encoder(encoder_name: str) -> tuple[nn.Module, int, int]:
    """Carga el encoder ViT preentrenado según el nombre de backbone especificado.

    Soporta tanto DINOv2 con registros (``dinov2reg_vit_*``) como DINOv2 sin
    registros (``dinov2_vit_*``), descargando los pesos desde el hub oficial
    de facebookresearch/dinov2.

    Args:
        encoder_name: Nombre del backbone. Debe ser una clave de
            :data:`ENCODER_CONFIGS`.
            Ejemplos: ``'dinov2_vit_base_14'``, ``'dinov2reg_vit_large_14'``.

    Returns:
        Tuple ``(encoder_module, embed_dim, num_heads)``.

    Raises:
        ValueError: Si ``encoder_name`` no está en :data:`ENCODER_CONFIGS`.
    """
    if encoder_name not in ENCODER_CONFIGS:
        available = list(ENCODER_CONFIGS.keys())
        msg = (
            f"Backbone '{encoder_name}' no reconocido. "
            f"Opciones disponibles: {available}"
        )
        raise ValueError(msg)

    cfg = ENCODER_CONFIGS[encoder_name]
    logger.info(
        "Cargando encoder '%s' desde %s/%s",
        encoder_name,
        cfg["hub_repo"],
        cfg["hub_name"],
    )

    encoder = torch.hub.load(
        cfg["hub_repo"],
        cfg["hub_name"],
        pretrained=True,
    )
    # El encoder permanece congelado; solo se entrenan bottleneck y decoder.
    encoder.requires_grad_(False)

    return encoder, cfg["embed_dim"], cfg["num_heads"]


def _get_default_target_layers(encoder_name: str) -> list[int]:
    """Devuelve las capas target por defecto según el backbone.

    Args:
        encoder_name: Clave de :data:`ENCODER_CONFIGS`.

    Returns:
        Lista de índices de capas del encoder.
    """
    cfg = ENCODER_CONFIGS[encoder_name]
    if cfg["target_layers_large"] is not None:
        return cfg["target_layers_large"]
    return cfg["target_layers_default"]


# ─────────────────────────────────────────────────────────────────────────────
# Componentes del modelo (tomados directamente de la implementación Anomalib)
# ─────────────────────────────────────────────────────────────────────────────

class LinearAttention2(nn.Module):
    """Atención lineal que no puede hacer focus, esencial para Dinomaly."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = eps
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Normalización ELU para atención lineal (evita el colapso de identidad)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Kernel trick: O(N*d^2) en vez de O(N^2*d)
        k_sum = k.sum(dim=-2)                                      # (B, H, d)
        denom = torch.einsum("bhnd,bhd->bhn", q, k_sum) + self.eps  # (B, H, N)
        kv = torch.einsum("bhnd,bhnv->bhdv", k, v)                # (B, H, d, d)
        numer = torch.einsum("bhnd,bhdv->bhnv", q, kv)            # (B, H, N, d)

        attn_out = numer / denom.unsqueeze(-1)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        return self.proj(attn_out)


class bMlp(nn.Module):  # noqa: N801
    """MLP con dropout usado como bottleneck."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VitDecoderBlock(nn.Module):
    """Bloque Transformer para el decoder de Dinomaly con LinearAttention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = partial(nn.LayerNorm, eps=1e-8),  # type: ignore[assignment]
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LinearAttention2(dim, num_heads=num_heads, qkv_bias=qkv_bias, eps=eps)
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = bMlp(dim, mlp_hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal: DinomalyModel
# ─────────────────────────────────────────────────────────────────────────────

class DinomalyModel(nn.Module):
    """DinomalyModel: detección de anomalías basada en Vision Transformer.

    Arquitectura encoder-bottleneck-decoder con soporte extendido de backbone.
    Ahora soporta DINOv2 con registros (``dinov2reg_vit_*``) **y sin registros**
    (``dinov2_vit_*``).

    Args:
        encoder_name: Backbone a usar. Debe ser una clave de
            :data:`ENCODER_CONFIGS`. Valor por defecto: ``'dinov2reg_vit_base_14'``.
        bottleneck_dropout: Dropout en el MLP bottleneck. Default: ``0.2``.
        decoder_depth: Número de bloques Transformer en el decoder. Default: ``8``.
        target_layers: Índices de capas del encoder de las que extraer
            características. Si ``None``, se calculan automáticamente.
        fuse_layer_encoder: Agrupaciones de capas encoder para fusión.
            Default: ``[[0, 1, 2, 3], [4, 5, 6, 7]]``.
        fuse_layer_decoder: Agrupaciones de capas decoder para fusión.
            Default: ``[[0, 1, 2, 3], [4, 5, 6, 7]]``.
        remove_class_token: Si ``True``, elimina el class token. Default: ``False``.
        use_context_recentering: Si ``True``, aplica Context-Aware Recentering
            (Dinomaly2). Default: ``False``.

    Example:
        >>> # Backbone original (con registros)
        >>> model = DinomalyModel(encoder_name="dinov2reg_vit_base_14")

        >>> # NUEVO: Backbone sin registros
        >>> model = DinomalyModel(encoder_name="dinov2_vit_base_14")

        >>> # NUEVO: ViT-Large sin registros
        >>> model = DinomalyModel(
        ...     encoder_name="dinov2_vit_large_14",
        ...     decoder_depth=8,
        ...     bottleneck_dropout=0.2,
        ... )
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        bottleneck_dropout: float = 0.2,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
        use_context_recentering: bool = False,
    ) -> None:
        super().__init__()

        self.encoder_name = encoder_name
        self.remove_class_token = remove_class_token
        self.use_context_recentering = use_context_recentering

        # ── Cargar encoder ──────────────────────────────────────────────────
        self.encoder, embed_dim, num_heads = _load_encoder(encoder_name)

        # ── Target layers ────────────────────────────────────────────────────
        if target_layers is None:
            target_layers = _get_default_target_layers(encoder_name)
        self.target_layers = target_layers

        # ── Fusión de capas ──────────────────────────────────────────────────
        self.fuse_layer_encoder = fuse_layer_encoder or [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder = fuse_layer_decoder or [[0, 1, 2, 3], [4, 5, 6, 7]]

        # ── Bottleneck: MLP ruidoso ──────────────────────────────────────────
        self.bottleneck = nn.ModuleList([
            bMlp(embed_dim, embed_dim * 4, embed_dim, drop=bottleneck_dropout),
        ])

        # ── Decoder: bloques Transformer con LinearAttention ─────────────────
        self.decoder = nn.ModuleList([
            VitDecoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(decoder_depth)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializa los pesos del bottleneck y decoder (truncated normal)."""
        trainable = nn.ModuleList([self.bottleneck, self.decoder])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    # ── Extracción de características del encoder ────────────────────────────

    def _extract_encoder_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extrae características de las capas intermedias del encoder DINOv2.

        Funciona tanto con modelos con registros (``dinov2reg``) como sin ellos
        (``dinov2``), usando la API ``get_intermediate_layers`` de DINOv2.

        Args:
            x: Batch de imágenes ``(B, C, H, W)``.

        Returns:
            Lista de tensores de características, uno por capa en
            ``self.target_layers``. Cada tensor tiene forma ``(B, N, D)``
            donde N son los tokens de parche.
        """
        # get_intermediate_layers devuelve una lista de tensores por cada capa.
        # return_class_token=True incluye el class token como primer elemento.
        features = self.encoder.get_intermediate_layers(
            x,
            n=self.target_layers,
            return_class_token=True,
        )
        # features es lista de tuplas (patch_tokens, class_token) o
        # simplemente tensores según la versión de DINOv2.
        # Normalizamos la salida para manejar ambos casos.
        patch_features = []
        cls_tokens = []

        for feat in features:
            if isinstance(feat, tuple):
                # API moderna: (patch_tokens, class_token)
                patch_tok, cls_tok = feat
            else:
                # API anterior: tensor (B, N+1, D) con class token al inicio
                cls_tok = feat[:, 0]
                patch_tok = feat[:, 1:]

            if self.remove_class_token:
                pass  # no añadimos cls_tok
            elif self.use_context_recentering:
                # Context-Aware Recentering (Dinomaly2): restar cls token
                patch_tok = patch_tok - cls_tok.unsqueeze(1)

            patch_features.append(patch_tok)
            cls_tokens.append(cls_tok)

        return patch_features

    # ── Fusión de capas ──────────────────────────────────────────────────────

    @staticmethod
    def _fuse_layers(
        features: list[torch.Tensor],
        fuse_groups: list[list[int]],
    ) -> list[torch.Tensor]:
        """Agrupa y promedia características de capas según los índices dados.

        Args:
            features: Lista de tensores ``(B, N, D)`` por capa.
            fuse_groups: Agrupaciones de índices, e.g. ``[[0,1,2,3], [4,5,6,7]]``.

        Returns:
            Lista de tensores fusionados, uno por grupo.
        """
        fused = []
        for group in fuse_groups:
            group_feats = torch.stack([features[i] for i in group], dim=0)
            fused.append(group_feats.mean(dim=0))
        return fused

    # ── Forward principal ────────────────────────────────────────────────────

    def get_encoder_decoder_outputs(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Extrae y procesa características a través del encoder y el decoder.

        Args:
            x: Imágenes de entrada ``(B, C, H, W)``.

        Returns:
            Tupla ``(encoder_feats, decoder_feats)`` donde cada elemento es
            una lista de tensores fusionados por grupo de capas.
        """
        # 1. Encoder: extraer características de las capas target
        enc_layer_feats = self._extract_encoder_features(x)

        # 2. Bottleneck: pasar solo las capas centrales (middle) por el MLP
        #    (siguiendo la implementación original de Dinomaly)
        bottleneck_input = torch.stack(enc_layer_feats, dim=0).mean(dim=0)  # (B, N, D)
        bottleneck_out = self.bottleneck[0](bottleneck_input)  # (B, N, D)

        # 3. Decoder: pasar la salida del bottleneck por los bloques Transformer
        dec_out = bottleneck_out
        decoder_layer_feats = []
        for block in self.decoder:
            dec_out = block(dec_out)
            decoder_layer_feats.append(dec_out)

        # 4. Fusionar capas del encoder y del decoder
        encoder_fused = self._fuse_layers(enc_layer_feats, self.fuse_layer_encoder)
        decoder_fused = self._fuse_layers(decoder_layer_feats, self.fuse_layer_decoder)

        return encoder_fused, decoder_fused

    @staticmethod
    def calculate_anomaly_maps(
        source_feature_maps: list[torch.Tensor],
        target_feature_maps: list[torch.Tensor],
        out_size: int | tuple[int, int] = 392,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Calcula mapas de anomalía comparando características encoder/decoder.

        Args:
            source_feature_maps: Características del encoder (lista de tensores).
            target_feature_maps: Características del decoder (lista de tensores).
            out_size: Tamaño de salida del mapa de anomalía.

        Returns:
            Tupla ``(anomaly_map, anomaly_map_list)`` donde ``anomaly_map``
            es el promedio de todas las escalas.
        """
        anomaly_map_list = []

        if isinstance(out_size, int):
            out_size = (out_size, out_size)

        for src, tgt in zip(source_feature_maps, target_feature_maps):
            # Calcular similitud coseno entre características encoder y decoder
            # src, tgt: (B, N, D)
            B, N, D = src.shape
            h = w = int(N ** 0.5)

            # Normalizar a esfera unitaria
            src_norm = F.normalize(src, dim=-1)
            tgt_norm = F.normalize(tgt, dim=-1)

            # Similitud coseno por token: (B, N)
            cos_sim = (src_norm * tgt_norm).sum(dim=-1)

            # Convertir a mapa espacial (B, 1, h, w)
            cos_map = cos_sim.reshape(B, 1, h, w)

            # Anomalía = 1 - similitud (mayor diferencia → mayor anomalía)
            anomaly_map = 1 - cos_map

            # Interpolar al tamaño de salida
            anomaly_map = F.interpolate(
                anomaly_map,
                size=out_size,
                mode="bilinear",
                align_corners=False,
            )
            anomaly_map_list.append(anomaly_map)

        # Combinar mapas de todas las escalas
        combined = torch.stack(anomaly_map_list, dim=0).mean(dim=0)  # (B, 1, H, W)
        return combined, anomaly_map_list

    def forward(
        self,
        batch: torch.Tensor,
        global_step: int | None = None,
    ) -> dict | "InferenceBatch":
        """Forward pass del modelo Dinomaly.

        Durante el entrenamiento devuelve un dict con características encoder
        y decoder para el cálculo de la pérdida. Durante la inferencia devuelve
        un ``InferenceBatch`` con mapas de anomalía y puntuaciones.

        Args:
            batch: Imágenes de entrada ``(B, C, H, W)``.
            global_step: Paso de entrenamiento actual (usado para la pérdida
                progresiva).

        Returns:
            Dict con ``'encoder_features'`` y ``'decoder_features'`` durante
            el entrenamiento, o ``InferenceBatch`` durante la inferencia.
        """
        encoder_features, decoder_features = self.get_encoder_decoder_outputs(batch)

        if self.training:
            return {
                "encoder_features": encoder_features,
                "decoder_features": decoder_features,
            }

        # ── Inferencia ───────────────────────────────────────────────────────
        # Tamaño de salida = tamaño de la imagen de entrada
        out_size = (batch.shape[-2], batch.shape[-1])
        anomaly_map, _ = self.calculate_anomaly_maps(
            encoder_features, decoder_features, out_size=out_size
        )

        # Puntuación de imagen = máximo del mapa de anomalía
        pred_score = anomaly_map.amax(dim=(-1, -2, -3))  # (B,)

        # Importación tardía para evitar dependencias circulares
        from anomalib.data.dataclasses.torch import InferenceBatch  # noqa: PLC0415

        return InferenceBatch(
            pred_score=pred_score,
            anomaly_map=anomaly_map.squeeze(1),  # (B, H, W)
        )
