public int read(byte[] buf, int offset, int numToRead) throws IOException {
	int totalRead = 0;

    if (hasHitEOF || entryOffset >= entrySize) {
        return -1;
    }

    if (currEntry == null) {
        throw new IllegalStateException("No current tar entry");
    }

    numToRead = Math.min(numToRead, available());
    
    totalRead = is.read(buf, offset, numToRead);
    count(totalRead);
    
    if (totalRead == -1) {
        hasHitEOF = true;
    } else {
        entryOffset += totalRead;
    }

    return totalRead;
}

// End


// Buggy Function
public Node findDefinition(String name) {
  Node n = scopes.get(0).getVar(name);
  if (n != null) {
    return n;
  }
  for (Node scope : scopes) {
    n = scope.getProp(name);
    if (n != null) {
      return n;
    }
  }
  return null;
}