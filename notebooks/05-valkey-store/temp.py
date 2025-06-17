# %%
import numpy as np
import json
import ast
import time
from typing import List, Dict, Any, Optional
from redis.commands.search.field import VectorField, TextField, TagField
from redis.commands.search.index_definition import (
    IndexDefinition,
    IndexType,
)  # Note: index_definition (lowercase)
from redis.commands.search.query import Query

#
