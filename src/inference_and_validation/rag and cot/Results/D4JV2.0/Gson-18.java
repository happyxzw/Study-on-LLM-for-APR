static Type getSupertype(Type context, Class<?> contextRawType, Class<?> supertype) {
  // wildcards are useless for resolving supertypes. As the upper bound has the same raw type, use it instead
  checkArgument(supertype.isAssignableFrom(contextRawType), "Supertype %s not a valid superclass of %s", supertype, contextRawType);
  return resolve(context, contextRawType,
      $Gson$Types.getGenericSupertype(context, contextRawType, supertype));
}